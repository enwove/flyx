# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""A .flyx file with a DV drafter (python/convert --dv) or a V-LoRA (--lora, or beside the DV) against what it was
made from.

    python -I tests/python/convert/check_dv_convert.py MODEL.flyx EXPORT_DIR [--canvas CANVAS.pt --selector SEL.pt]
        [--vlora VLORA.pt] [--hf HF_DIR]

* Its dv.* and dv.sel.* tensors are EXPORT_DIR's canvas.safetensors and selector.safetensors (export_dv.py's, or
  the earlier CUDA port's), name for name, byte for byte, in their dtypes; and it has no other DV tensor.
* Its header's dv section says what EXPORT_DIR's dv.json says (slots, tree, fill, ranks, causal, the mask token).
* The DV reference (python/reference/flyx_ref/dv.py) reads the file's section as the DvConfig a run of dv.json's
  settings uses (from_flyx against DvConfig.set_tree), every field; and the section of the other tree kind too.
* --canvas, --selector (bifill-dv's checkpoints, read with torch.load(weights_only=True)): the reference loads the
  file to the weights it loads from them (from_flyx against canvas_from_pt, selector_from_pt), byte for byte.
* --hf: and the target's weights and config to those of the Hugging Face directory (hf_llama).
* A V-LoRA (EXPORT_DIR's vlora.safetensors, export_dv.py --vlora's): the file's target.L{l}.{attn,mlp}.*.lora_{a,b}
  are its L{l}.*.lora_{a,b}, byte for byte, and no other adapter; its header's lora section is vlora.json's (rank,
  scale, layers, source); the reference reads the scale (LlamaConfig.lora_scale); --vlora (bifill-dv's checkpoint):
  the reference loads the file to the adapters it loads from it (llama_from_flyx against vlora_from_pt).  The export
  without dv.json: a file of a V-LoRA alone (no DV section, no dv.* tensor).

ctests (dv_convert, lora_convert) run it on the tiny Llama's DV and V-LoRA files (make_tiny.py); by hand on
MiniCPM5-1B's (docs/design/dv.md).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python", "convert"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python", "reference"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))
from flyx_convert.dv import parse_tree, read_meta  # noqa: E402
from flyx_convert.fileformat import Reader  # noqa: E402
from flyx_ref import dv  # noqa: E402
from check import check, finish  # noqa: E402


def same_bytes(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8),
                                                                     b.contiguous().view(torch.uint8))


def compare(what, mine, theirs):
    """mine's tensors against theirs, by name: all of theirs, byte for byte, and none besides."""
    same = sum(k in mine and same_bytes(mine[k], v) for k, v in theirs.items())
    extra = sorted(set(mine) - set(theirs))
    dtypes = sorted({str(v.dtype).removeprefix("torch.") for v in theirs.values()})
    check(same == len(theirs) and not extra, f"{what}: {same}/{len(theirs)} tensors byte for byte ({', '.join(dtypes)})"
          + (f"; besides them: {extra[:5]}" if extra else ""))


def same_config(what, got, want):
    differ = {f.name: (getattr(got, f.name), getattr(want, f.name)) for f in dataclasses.fields(want)
              if getattr(got, f.name) != getattr(want, f.name)}
    check(not differ, what + ("" if not differ else f"; read, wanted: {differ}"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("export")
    ap.add_argument("--canvas")
    ap.add_argument("--selector")
    ap.add_argument("--vlora")
    ap.add_argument("--hf")
    a = ap.parse_args(argv)
    from safetensors.torch import load_file
    r = Reader(a.model)
    h = r.header
    with_dv = os.path.exists(os.path.join(a.export, "dv.json"))
    lora = vlora(a, r, h, load_file)
    if not with_dv:
        check(not h.get("dv") and not any(n.startswith("dv.") for n in h["tensors"]),
              f"{os.path.basename(a.model)}: a V-LoRA alone (no DV section, no dv.* tensor)")
        cfg, W = dv.llama_from_flyx(a.model)
        target_config(a, cfg, W, lora)
        return finish("check_dv_convert")
    mine = {n: r.torch(n) for n in h["tensors"] if n.startswith("dv.")}
    export = {"dv." + k: v for k, v in load_file(os.path.join(a.export, "canvas.safetensors")).items()}
    export.update({"dv.sel." + k: v for k, v in load_file(os.path.join(a.export, "selector.safetensors")).items()})
    compare(f"{os.path.basename(a.model)}'s DV tensors against {a.export}", mine, export)

    meta, d = read_meta(a.export), h["dv"]
    want = {"slots": meta["slots"], "tree": parse_tree(meta["tree"], meta["slots"], meta["topk"]),
            "topk": meta["topk"], "res_k": meta["res_k"], "fill": meta["fill"], "causal": meta["causal"],
            "rank": meta["rank"], "sel_rank": meta["selector"]["rank"], "temp": meta.get("temp", 1.0)}
    got = {k: d[k] for k in want}
    check(got == want, f"header dv section: {json.dumps(got)}" + ("" if got == want else f", dv.json: {want}"))
    mask = meta["mask_token"]
    added = {t["content"]: t["id"] for t in json.loads(r.blob("tokenizer.added"))}
    mask_id = added.get(mask) if isinstance(mask, str) else mask
    check(d["mask_token"] == mask_id, f"mask token {mask!r} -> {d['mask_token']}")

    def run_config(spec):  # what python/reference/dv_gsm8k.py runs with dv.json's settings and this tree
        return dv.DvConfig(slots=meta["slots"], mask_token=mask_id, topk=meta["topk"], fill=meta["fill"],
                           causal=bool(meta["causal"]), temp=meta.get("temp", 1.0)).set_tree(spec)

    cfg, W, dvc = dv.from_flyx(a.model)
    same_config(f"the reference's DvConfig of the file: {dvc}", dvc, run_config(meta["tree"]))
    other = "pool:7:3:1" if d["tree"]["kind"] == "beam" else "beam:7:4:cov"
    section = {**d, "tree": parse_tree(other, d["slots"], d["topk"])}
    same_config(f"the reference's DvConfig of the section with tree {other}", dv.DvConfig.from_header(section),
                run_config(other))
    if a.canvas and a.selector:
        canvas, _ = dv.canvas_from_pt(a.canvas, cfg.layers)
        compare("from_flyx against canvas_from_pt", {k: v for k, v in W.items() if k.startswith("dv.") and
                                                     not k.startswith("dv.sel.")}, canvas)
        compare("from_flyx against selector_from_pt", {k: v for k, v in W.items() if k.startswith("dv.sel.")},
                dv.selector_from_pt(a.selector))
    target_config(a, cfg, W, lora)
    return finish("check_dv_convert")


def vlora(a, r, h, load_file):
    """The file's adapters against EXPORT_DIR's V-LoRA (vlora.safetensors, vlora.json), when it has one: byte for byte,
    the header's lora section, the file's format 2 requiring it (which a build of format 1 only refuses); None, none
    in the file and format 1 when it has none."""
    mine = {n: r.torch(n) for n in h["tensors"] if n.startswith("target.") and ".lora_" in n}
    if not os.path.exists(os.path.join(a.export, "vlora.json")):
        check(not mine and not h.get("lora") and r.version == 1 and not r.requires,
              f"{os.path.basename(a.model)}: no V-LoRA, as {a.export}; format {r.version}, requires {r.requires}")
        return None
    check(r.version == 2 and r.requires == ["lora"],
          f"{os.path.basename(a.model)}: format {r.version}, requires {r.requires} (2, ['lora']: its adapters)")
    from flyx_convert.lora import BLOCK, read_lora_meta
    meta = read_lora_meta(a.export)
    theirs = {}
    for k, v in load_file(os.path.join(a.export, "vlora.safetensors")).items():
        layer, linear, part = k.split(".")
        theirs[f"target.{layer}.{BLOCK[linear]}.{linear}.{part}"] = v
    compare(f"{os.path.basename(a.model)}'s V-LoRA tensors against {a.export}", mine, theirs)
    got = {k: h.get("lora", {}).get(k) for k in ("rank", "scale", "layers", "source")}
    want = {k: meta.get(k) for k in ("rank", "scale", "layers", "source")}
    check(got == want, f"header lora section: {json.dumps(got)}" + ("" if got == want else f", vlora.json: {want}"))
    return meta


def target_config(a, cfg, W, lora):
    """The reference's reading of the target: its V-LoRA's scale, its adapters against --vlora's checkpoint, its
    weights and config against --hf's directory."""
    scale = lora["scale"] if lora else 1.0
    check(cfg.lora_scale == scale, f"the reference's V-LoRA scale: {cfg.lora_scale} (the file's {scale})")
    if a.vlora:
        compare("llama_from_flyx against vlora_from_pt (the target's own adapters)",
                {k: v for k, v in W.items() if k.startswith("target.") and ".lora_" in k},
                dv.vlora_from_pt(a.vlora, cfg.layers))
    if a.hf:
        hcfg, target = dv.hf_llama(a.hf)
        same = dataclasses.replace(cfg, lora_scale=hcfg.lora_scale) == hcfg
        check(same, f"config: {cfg}" + ("" if same else f", the Hugging Face directory's: {hcfg}"))
        compare("the file against hf_llama (the target, its adapters left out)",
                {k: v for k, v in W.items() if k.startswith("target.") and ".lora_" not in k}, target)


if __name__ == "__main__":
    sys.exit(main())
