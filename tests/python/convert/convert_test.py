# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""python/convert: what the converter refuses rather than write a model the engine computes differently from
transformers, and the tokens a tokenizer's post-processor puts before a text (the engine puts them before a raw
prompt); a DV drafter's export (export_dv.py's directory, the earlier CUDA port's too) and what is refused of it and
of bifill-dv's checkpoints; a V-LoRA's (the target's own adapters: export_dv.py --vlora's directory) and what is
refused of it and of bifill-dv's raft.py checkpoints.  No model, no torch: configs written here, a tokenizer trained
here, safetensors written here, checkpoints as numpy arrays.

    python tests/python/convert/convert_test.py

The DV tensors of a converted file against their sources, byte for byte: check_dv_convert.py.
"""
from __future__ import annotations

import json
import os
import re
import struct
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python", "convert"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))

from check import check, finish  # noqa: E402
import export_dv  # noqa: E402
from flyx_convert.convert import (canonical_dense_config, canonical_draft_config,  # noqa: E402
                                  canonical_target_config, convert)
from flyx_convert import eagle4  # noqa: E402
from flyx_convert.dv import add_dv, parse_tree, pool_size  # noqa: E402
from flyx_convert.fileformat import MAGIC, Reader, Writer  # noqa: E402
from flyx_convert.lora import add_lora  # noqa: E402
from flyx_convert.tokenizer_export import MINICPM5_NUMBERS, MINICPM5_SPLIT, export  # noqa: E402


def refused(f, arg, because: str) -> bool:
    try:
        f(arg)
    except ValueError as e:
        return because in str(e)
    return False


# MiniCPM5-1B's config.json, as published (a Llama).
LLAMA = {"model_type": "llama", "hidden_size": 1536, "num_hidden_layers": 24, "num_attention_heads": 16,
         "num_key_value_heads": 2, "head_dim": 128, "intermediate_size": 4608, "vocab_size": 130560,
         "rms_norm_eps": 1e-6, "rope_theta": 5000000, "rope_scaling": None, "hidden_act": "silu",
         "attention_bias": False, "mlp_bias": False, "tie_word_embeddings": False}
# Qwen3.5's text config, its rotary in rope_parameters.
QWEN35 = {"hidden_size": 2560, "num_hidden_layers": 4, "num_attention_heads": 16, "num_key_value_heads": 4,
          "head_dim": 256, "intermediate_size": 9216, "vocab_size": 248320, "rms_norm_eps": 1e-6,
          "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                              "mrope_section": [11, 11, 10], "mrope_interleaved": True},
          "linear_num_key_heads": 16, "linear_num_value_heads": 32, "linear_key_head_dim": 128,
          "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4}


def rotary():
    """A scaled rotary is refused whichever key names its kind (transformers reads `rope_type` and the older `type`)
    and wherever it is (rope_parameters, the older rope_scaling): the engine's rotary is unscaled."""
    check(canonical_dense_config(LLAMA)["rope_theta"] == 5e6, "MiniCPM5-1B's config: plain rotary, theta 5e6")
    for scaling in ({"rope_type": "default"}, {"type": "default"}, {"rope_type": "default", "factor": 1.0}):
        check(canonical_dense_config({**LLAMA, "rope_scaling": scaling})["rope_theta"] == 5e6,
              f"rope_scaling {scaling}: taken as plain")
    for kind in ("linear", "dynamic", "yarn", "llama3"):
        for key in ("rope_type", "type"):
            scaling = {key: kind, "factor": 4.0}
            check(refused(canonical_dense_config, {**LLAMA, "rope_scaling": scaling}, f"unsupported rope type {kind}"),
                  f"rope_scaling {scaling}: refused")
    check(refused(canonical_dense_config, {**LLAMA, "rope_scaling": {"factor": 2.0}}, "scaled rotary"),
          "a factor without a kind: refused")
    check(refused(canonical_dense_config, {**LLAMA, "rope_scaling": {"type": "default", "factor": 2.0}},
                  "scaled rotary"), "a factor of 2 called default: refused")
    both = {**LLAMA, "rope_parameters": {"rope_type": "default", "rope_theta": 5e6},
            "rope_scaling": {"type": "linear", "factor": 2.0}}
    check(refused(canonical_dense_config, both, "unsupported rope type linear"),
          "plain rope_parameters beside a scaled legacy rope_scaling: refused")
    check(canonical_target_config(QWEN35)["rot_dim"] == 64, "Qwen3.5's config: plain rotary on a quarter of the head")
    plain = {k: v for k, v in QWEN35["rope_parameters"].items() if k != "rope_type"}
    for key in ("rope_type", "type"):
        cfg = {**QWEN35, "rope_parameters": {**plain, key: "yarn", "factor": 4.0}}
        check(refused(canonical_target_config, cfg, "unsupported rope type yarn"),
              f"Qwen3.5 with {key} yarn: refused")


def tokenizer_json(path: str, special_s: bool, post_processor) -> str:
    """A small byte-level BPE with MiniCPM5's pre-tokenizer; <s> special or not, the post-processor given."""
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split(Regex(r), behavior="isolated", invert=False) for r in (MINICPM5_NUMBERS, MINICPM5_SPLIT)]
        + [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=["</s>"] + (["<s>"] if special_s else []),
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(["a small text, 123 4567 to train on"] * 4, trainer)
    if not special_s:
        tok.add_tokens(["<s>"])
    tj = json.loads(tok.to_str())
    tj["post_processor"] = post_processor(tok.token_to_id("<s>"), tok.token_to_id("</s>"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tj, f)
    return path


def template(single, ids):
    return {"type": "TemplateProcessing", "single": single, "pair": single,
            "special_tokens": {name: {"id": name, "ids": [i], "tokens": [name]} for name, i in ids.items()}}


def special(name):
    return {"SpecialToken": {"id": name, "type_id": 0}}


def text(name="A"):
    return {"Sequence": {"id": name, "type_id": 0}}


def post_processor(tmp: str):
    """The post-processor's tokens before a text become the header's prefix_ids; anything else is refused."""
    byte_level = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
    accepted = {
        "none": (lambda s, e: None, lambda s, e: None),
        "ByteLevel (Qwen's)": (lambda s, e: byte_level, lambda s, e: None),
        "<s> $A (MiniCPM5's)": (lambda s, e: template([special("<s>"), text()], {"<s>": s}), lambda s, e: [s]),
        "ByteLevel, then <s> $A": (lambda s, e: {"type": "Sequence", "processors": [
            byte_level, template([special("<s>"), text()], {"<s>": s})]}, lambda s, e: [s]),
        "$A alone": (lambda s, e: template([text()], {}), lambda s, e: None),
    }
    for name, (make, want) in accepted.items():
        info, _ = export(tokenizer_json(os.path.join(tmp, "t.json"), True, make))
        s = json.load(open(os.path.join(tmp, "t.json"), encoding="utf-8"))["added_tokens"]
        ids = {a["content"]: a["id"] for a in s}
        check(info.get("prefix_ids") == want(ids["<s>"], ids["</s>"]), f"post-processor {name}: prefix_ids "
              f"{info.get('prefix_ids')}")
    refusals = {
        "a token after the text": lambda s, e: template([special("<s>"), text(), special("</s>")],
                                                        {"<s>": s, "</s>": e}),
        "the second text in the single form": lambda s, e: template([special("<s>"), text("B")], {"<s>": s}),
        "two templates": lambda s, e: {"type": "Sequence", "processors": [
            template([special("<s>"), text()], {"<s>": s}), template([special("<s>"), text()], {"<s>": s})]},
        "BERT's": lambda s, e: {"type": "BertProcessing", "sep": ["</s>", e], "cls": ["<s>", s]},
        "RoBERTa's": lambda s, e: {"type": "RobertaProcessing", "sep": ["</s>", e], "cls": ["<s>", s],
                                   "trim_offsets": True, "add_prefix_space": False},
    }
    for name, make in refusals.items():
        check(refused(export, tokenizer_json(os.path.join(tmp, "t.json"), True, make), "post_processor"),
              f"post-processor with {name}: refused")
    not_special = tokenizer_json(os.path.join(tmp, "t.json"), False,
                                 lambda s, e: template([special("<s>"), text()], {"<s>": s}))
    check(refused(export, not_special, "not all special"), "a prefix token that is not special: refused")


def save_safetensors(path: str, tensors: dict):
    """{name: (dtype "BF16" | "F32", array)} as a safetensors file (bf16: the array's uint16 bits)."""
    header, blobs, off = {}, [], 0
    for name, (dtype, a) in tensors.items():
        b = np.ascontiguousarray(a).tobytes()
        header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))


# A Llama's canonical config, small: what a DV export must fit.
TINY = {"hidden": 8, "layers": 2, "n_heads": 2, "n_kv": 1, "head_dim": 4, "inter": 16, "vocab": 50}
MASK = "<mask>"


def dv_export(d: str, meta_changes=None, drop=(), extra=None, retype=None, reshape=None, legacy=False):
    """An export directory for TINY (rank 4, selector rank 3, 6 slots), changed as asked."""
    rng = np.random.default_rng(0)
    D, q, kv, inter, R = 8, 8, 4, 16, 4
    dims = {"q": (q, D), "k": (kv, D), "v": (kv, D), "o": (D, q), "gate": (inter, D), "up": (inter, D),
            "down": (D, inter)}
    bf = lambda *s: ("BF16", rng.integers(0, 1 << 15, size=s, dtype=np.uint16))  # noqa: E731
    canvas = {"fnormc": bf(D), "mask_emb": bf(D), "res_in": bf(2 * D, D), "res_out": bf(D, D)}
    for layer in range(2):
        canvas[f"L{layer}.ln1c"], canvas[f"L{layer}.ln2c"] = bf(D), bf(D)
        for n, (o, i) in dims.items():
            canvas[f"L{layer}.{n}.lora_a"], canvas[f"L{layer}.{n}.lora_b"] = bf(R, i), bf(o, R)
    sel = {n: ("F32", rng.standard_normal((3, D)).astype(np.float32)) for n in ("A", "H", "B")}
    sel["alpha"] = ("F32", np.array([0.85], dtype=np.float32))
    for name in drop:
        canvas.pop(name, None)
        sel.pop(name, None)
    canvas.update(extra or {})
    for name, dtype in (retype or {}).items():
        canvas[name] = (dtype, canvas[name][1].astype(np.float32) if dtype == "F32" else canvas[name][1])
    for name, shape in (reshape or {}).items():
        canvas[name] = bf(*shape)
    os.makedirs(d, exist_ok=True)
    save_safetensors(os.path.join(d, "canvas.safetensors"), canvas)
    if "selector" not in drop:
        save_safetensors(os.path.join(d, "selector.safetensors"), sel)
    meta = {"kind": "dv", "layers": 2, "slots": 6, "rank": R, "res_k": 16, "causal": False, "fill": 0.9, "topk": 16,
            "tree": "beam:7:2:cov", "mask_token": MASK, "selector": {"rank": 3}, "source": {"canvas": "c.pt"}}
    if legacy:  # the earlier CUDA port's words, its pool tree
        meta = {k: v for k, v in meta.items() if k not in ("slots", "causal", "tree")}
        meta.update({"rows": 6, "canvas_causal": False, "tree": 7, "tree_r": 3, "tree_m": 1})
    meta.update(meta_changes or {})
    with open(os.path.join(d, "dv.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return d


def dv_drafter(tmp: str):
    """A DV export becomes dv.* and dv.sel.* (as stored) and the header's dv section; the earlier CUDA port's dv.json
    reads the same; everything that does not fit the model or cannot be drafted is refused."""
    added = {MASK: 49}

    def take(d):
        w = Writer()
        return add_dv(w, d, TINY, added, log=lambda *_: None), w

    def refused_dv(d, because):
        try:
            take(d)
        except (ValueError, FileNotFoundError) as e:
            return because in str(e)
        return False

    head, w = take(dv_export(os.path.join(tmp, "ok")))
    names = {e[1]: e[3]["dtype"] for e in w.entries}
    check(len(names) == 4 + 2 * 16 + 4 and all(v == "bf16" for k, v in names.items() if not k.startswith("dv.sel."))
          and all(names[f"dv.sel.{n}"] == "f32" for n in ("A", "H", "B", "alpha")),
          f"an export: {len(names)} tensors, the canvas bf16, the selector float32")
    check(head == {"slots": 6, "tree": {"kind": "beam", "width": 7, "beam": 2, "cov": True}, "topk": 16, "res_k": 16,
                   "fill": 0.9, "causal": False, "mask_token": 49, "rank": 4, "sel_rank": 3, "temp": 1.0,
                   "source": {"canvas": "c.pt"}}, f"the header's dv section: {head}")
    old, _ = take(dv_export(os.path.join(tmp, "legacy"), legacy=True))
    check(old["tree"] == {"kind": "pool", "width": 7, "ranks": 3} and old["slots"] == 6,
          f"the earlier CUDA port's dv.json: its pool tree, {old['tree']}")
    num, _ = take(dv_export(os.path.join(tmp, "id"), {"mask_token": 7}))
    check(num["mask_token"] == 7, "a mask token given by id")
    # Every setting the header carries is dv.json's, not a default: a causal canvas, another fill and temperature.
    set_, _ = take(dv_export(os.path.join(tmp, "set"), {"causal": True, "fill": 0.5, "temp": 0.5}))
    check(set_["causal"] is True and set_["fill"] == 0.5 and set_["temp"] == 0.5,
          f"causal, fill 0.5, temp 0.5: the header's ({set_['causal']}, {set_['fill']}, {set_['temp']})")
    cases = [
        ("layers", {"meta_changes": {"layers": 3}}, "a DV for 3 layers"),
        ("mask", {"meta_changes": {"mask_token": "<nope>"}}, "is not a token of this model"),
        ("mask id", {"meta_changes": {"mask_token": 50}}, "outside the vocabulary"),
        ("selector", {"drop": ("selector",)}, "no selector"),
        ("tensor", {"drop": ("L1.down.lora_b",)}, "missing DV tensors"),
        ("unknown", {"extra": {"spine.w": ("BF16", np.zeros(8, dtype=np.uint16))}}, "unknown DV tensor"),
        ("shape", {"reshape": {"L0.q.lora_a": (5, 8)}}, "has shape [5, 8]"),
        ("dtype", {"retype": {"L0.q.lora_a": "F32"}}, "is f32, not bf16"),
        # A slot's 16 candidates and z over those 16, bifill-dv's and the kernels' (kDvTop): no other count.
        ("res_k", {"meta_changes": {"res_k": 8}}, "keeps 16 candidates"),
        ("topk", {"meta_changes": {"topk": 32, "res_k": 32}}, "keeps 16 candidates"),
        ("topk 8", {"meta_changes": {"topk": 8, "res_k": 8}}, "keeps 16 candidates"),
        ("topk 1", {"meta_changes": {"topk": 1, "res_k": 1}}, "keeps 16 candidates"),
        ("selector of 8", {"meta_changes": {"topk": 8}}, "keeps 16 candidates"),
        ("fill", {"meta_changes": {"fill": 1.5}}, "not a probability"),
        ("slots", {"meta_changes": {"slots": 1}}, "at least 2"),
        ("tree", {"meta_changes": {"tree": "beam:31:2"}}, "holds fewer than 31 nodes"),
        ("kind", {"meta_changes": {"kind": "dflash2"}}, "not a DV export"),
    ]
    for name, change, because in cases:
        check(refused_dv(dv_export(os.path.join(tmp, name), **change), because), f"a DV export, {name}: refused")
    # The tree a round drafts: the converter's and export_dv.py's check, before anything is written.
    check(parse_tree("beam:31:8:cov", 32) == {"kind": "beam", "width": 31, "beam": 8, "cov": True} and
          parse_tree("pool:31:7:1", 32) == {"kind": "pool", "width": 31, "ranks": 7} and
          parse_tree("pool:31:7", 32) == parse_tree("pool:31:7:1", 32) and
          parse_tree("beam:15:8", 64)["cov"] is False, "trees beam:31:8:cov, pool:31:7:1 (m = 1 by default)")
    check(pool_size(31, 7) == 3503, "the pool of ST 31/7/1 has 3503 tuples")
    bad = {"tri:31:8": "beam:T:W[:cov] or pool:T:R[:1]", "beam:31": "beam:T:W[:cov] or pool:T:R[:1]",
           "beam:x:8": "beam:T:W[:cov] or pool:T:R[:1]", "beam:31:8:wide": "can only be cov",
           "beam:0:8": "at least 1", "beam:31:0": "at least 1", "pool:31:7:2": "once (m = 1)",
           "pool:31:16:1": "past a slot's 16 candidates", "beam:31:4:cov:x": "beam:T:W[:cov] or pool:T:R[:1]"}
    for spec, because in bad.items():
        check(refused(lambda t: parse_tree(t, 32), spec, because), f"tree {spec!r}: refused ({because})")
    # Too few slots for the tree: 2 depths of a pool of rank 1 hold 5 tuples, 15 depths of a beam of 1 15 nodes.
    check(refused(lambda t: parse_tree(t, 3), "pool:31:1:1", "a pool of 5 tuples cannot give 31 drafts"),
          "tree pool:31:1:1 on 3 slots: refused")
    check(refused(lambda t: parse_tree(t, 16), "beam:31:1", "holds fewer than 31 nodes"),
          "tree beam:31:1 on 16 slots: refused")
    check(refused(lambda t: parse_tree(t, 1), "beam:4:4", "at least 2"), "a canvas of one slot: refused")
    # A DV drafter is for a Llama: a Qwen3 checkpoint with one is refused before its weights are read.
    q3 = os.path.join(tmp, "qwen3")
    os.makedirs(q3, exist_ok=True)
    with open(os.path.join(q3, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**LLAMA, "model_type": "qwen3"}, f)
    check(refused(lambda d: convert(d, os.path.join(tmp, "x.flyx"), dv_dir=os.path.join(tmp, "ok")), q3,
                  "only supported for Llama"), "--dv with a Qwen3 checkpoint: refused")


def lora_export(d: str, meta_changes=None, drop=(), extra=None, retype=None, reshape=None):
    """An export_dv.py --vlora directory for TINY (rank 2), changed as asked."""
    rng = np.random.default_rng(1)
    D, q, kv, inter, R = 8, 8, 4, 16, 2
    dims = {"q": (q, D), "k": (kv, D), "v": (kv, D), "o": (D, q), "gate": (inter, D), "up": (inter, D),
            "down": (D, inter)}
    bf = lambda *s: ("BF16", rng.integers(0, 1 << 15, size=s, dtype=np.uint16))  # noqa: E731
    t = {}
    for layer in range(2):
        for n, (o, i) in dims.items():
            t[f"L{layer}.{n}.lora_a"], t[f"L{layer}.{n}.lora_b"] = bf(R, i), bf(o, R)
    for name in drop:
        t.pop(name)
    t.update(extra or {})
    for name, dtype in (retype or {}).items():
        t[name] = (dtype, t[name][1].astype(np.float32))
    for name, shape in (reshape or {}).items():
        t[name] = bf(*shape)
    os.makedirs(d, exist_ok=True)
    save_safetensors(os.path.join(d, "vlora.safetensors"), t)
    meta = {"kind": "lora", "layers": 2, "rank": R, "scale": 1.0, "source": {"vlora": "v.pt", "sha256": "ab" * 32}}
    meta.update(meta_changes or {})
    with open(os.path.join(d, "vlora.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return d


def vlora_files(tmp: str):
    """A V-LoRA export becomes the target's target.L{l}.{attn,mlp}.<linear>.lora_{a,b} (as stored, bf16) and the
    header's lora section; what does not fit the model is refused; --lora on a Qwen3.5 checkpoint, and beside a DV
    export holding a V-LoRA of its own, refused."""
    def take(d):
        w = Writer()
        return add_lora(w, d, TINY, log=lambda *_: None), w

    def refused_lora(d, because):
        try:
            take(d)
        except (ValueError, FileNotFoundError) as e:
            return because in str(e)
        return False

    head, w = take(lora_export(os.path.join(tmp, "lora_ok")))
    names = {e[1]: (e[3]["dtype"], e[3]["shape"]) for e in w.entries}
    check(len(names) == 2 * 14 and names["target.L0.attn.q.lora_a"] == ("bf16", [2, 8]) and
          names["target.L1.mlp.down.lora_b"] == ("bf16", [8, 2]) and
          names["target.L1.attn.o.lora_a"] == ("bf16", [2, 8]) and
          names["target.L0.mlp.down.lora_a"] == ("bf16", [2, 16]),
          f"a V-LoRA export: {len(names)} tensors, target.L{{l}}.{{attn,mlp}}.<linear>.lora_{{a,b}} in bf16")
    check(head == {"rank": 2, "scale": 1.0, "layers": 2, "source": {"vlora": "v.pt", "sha256": "ab" * 32}},
          f"the header's lora section: {head}")
    half, _ = take(lora_export(os.path.join(tmp, "lora_half"), {"scale": 0.5}))
    check(half["scale"] == 0.5, "the scale is vlora.json's")
    cases = [
        ("layers", {"meta_changes": {"layers": 3}}, "a V-LoRA for 3 layers"),
        ("rank in vlora.json", {"meta_changes": {"rank": 4}}, "rank 2 ([2, 8]), vlora.json says 4"),
        ("rank of one adapter", {"reshape": {"L1.v.lora_b": (4, 3)}}, "L1.v.lora_b: rank 3"),
        ("shape", {"reshape": {"L0.o.lora_a": (2, 16)}}, "L0.o.lora_a: shape [2, 16], the model needs [2, 8]"),
        ("missing", {"drop": ("L1.down.lora_b",)}, "missing V-LoRA tensors: ['L1.down.lora_b']"),
        ("unknown", {"extra": {"L0.gate.lora_c": ("BF16", np.zeros((2, 8), dtype=np.uint16))}},
         "unknown V-LoRA tensors"),
        ("dtype", {"retype": {"L0.q.lora_a": "F32"}}, "is f32, not bf16"),
        ("rank 0", {"meta_changes": {"rank": 0}}, "rank 0 (at least 1)"),
        ("scale 0", {"meta_changes": {"scale": 0.0}}, "a finite number other than 0"),
        ("scale not finite", {"meta_changes": {"scale": float("inf")}}, "a finite number other than 0"),
        ("kind", {"meta_changes": {"kind": "dv"}}, "not a V-LoRA export"),
    ]
    for name, change, because in cases:
        check(refused_lora(lora_export(os.path.join(tmp, "lora_" + name.replace(" ", "_")), **change), because),
              f"a V-LoRA export, {name}: refused")
    q35 = os.path.join(tmp, "qwen35")
    os.makedirs(q35, exist_ok=True)
    with open(os.path.join(q35, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**QWEN35, "model_type": "qwen3_5_text"}, f)
    check(refused(lambda d: convert(d, os.path.join(tmp, "x.flyx"), lora_dir=os.path.join(tmp, "lora_ok")), q35,
                  "only supported for Llama and Qwen3"), "--lora with a Qwen3.5 checkpoint: refused")
    both = dv_export(os.path.join(tmp, "dv_with_lora"))
    lora_export(both)
    llama = os.path.join(tmp, "llama")
    os.makedirs(llama, exist_ok=True)
    with open(os.path.join(llama, "config.json"), "w", encoding="utf-8") as f:
        json.dump(LLAMA, f)
    alone = os.path.join(tmp, "lora_ok")
    check(refused(lambda d: convert(llama, os.path.join(tmp, "x.flyx"), dv_dir=d, lora_dir=alone), both, "two V-LoRAs"),
          "--lora beside a DV export with a V-LoRA of its own: refused")


def file_format(tmp: str):
    """A file with parts a reader must not ignore (the header's requires: a V-LoRA's "lora") is format 2, which a
    reader of format 1 only refuses; one without them format 1.  The reader refuses requires in format 1, format 2
    without it, and requires that is not a list of names."""
    def written(meta, name):
        w = Writer()
        w.meta = meta
        w.add_tensor("x", "u8", [4], bytes(4))
        path = os.path.join(tmp, name)
        w.write(path)
        with open(path, "rb") as f:
            return path, struct.unpack("<I", f.read(12)[8:12])[0]

    plain, v1 = written({"arch": "llama", "lora": None}, "plain.flyx")
    lora, v2 = written({"arch": "llama", "lora": {"rank": 2}, "requires": ["lora"]}, "lora.flyx")
    r1, r2 = Reader(plain), Reader(lora)
    check(v1 == 1 and r1.version == 1 and r1.requires == [], f"a file requiring nothing: format {v1}")
    check(v2 == 2 and r2.version == 2 and r2.requires == ["lora"], f"a file requiring its adapters: format {v2}")
    for f in (r1, r2):
        f.mm.close()
        f.f.close()
    check(refused(lambda m: written(m, "bad.flyx"), {"arch": "llama", "requires": []}, "not a list of the parts"),
          "requires that names nothing: not written")

    def rewritten(version, header):  # a file of its own each: a refused one may stay mapped until collected
        hj = json.dumps(header).encode()
        path = os.path.join(tmp, f"rewritten{len(os.listdir(tmp))}.flyx")
        with open(path, "wb") as f:
            f.write(MAGIC + struct.pack("<IIQ", version, 0, len(hj)) + hj)
            f.write(bytes(4096 - 24 - len(hj)))
        return path

    def read_refused(path, because):
        try:
            r = Reader(path)
        except ValueError as e:
            return because in str(e)
        r.mm.close()
        r.f.close()
        return False

    base = {"arch": "llama", "tensors": {}, "blobs": {}}
    check(read_refused(rewritten(1, {**base, "requires": ["lora"]}), "format version 1 with 'requires'"),
          "format 1 with requires: refused")
    check(read_refused(rewritten(2, base), "format version 2 without 'requires'"), "format 2 without requires: refused")
    check(read_refused(rewritten(2, {**base, "requires": "lora"}), "not a list of the parts"),
          "format 2 with requires not a list: refused")
    check(read_refused(rewritten(3, {**base, "requires": ["lora"]}), "expected 1 or 2"), "format 3: refused")


def vlora_checkpoints():
    """export_dv.py's checks on bifill-dv's V-LoRA checkpoints (raft.py's, as loaded): whole layers of q, k, v, o,
    gate, up, down of one rank that fit one layer's linears, and no scale of their own."""
    D, q, kv, inter, R = 8, 8, 4, 16, 2
    dims = {"q": (q, D), "k": (kv, D), "v": (kv, D), "o": (D, q), "gate": (inter, D), "up": (inter, D),
            "down": (D, inter)}

    def ck(layers=2, rank=R, cfg=None, change=None):
        lora = [{"A": np.zeros((rank, dims[n][1])), "B": np.zeros((dims[n][0], rank))}
                for _ in range(layers) for n in dims]
        if change:
            change(lora)
        return {"lora": lora, "cfg": {"r": rank, **(cfg or {})}, "step": 20}

    check(export_dv.check_vlora(ck()) == (2, 2), "a V-LoRA of 2 layers, rank 2")
    check(export_dv.check_vlora(ck(3, 4)) == (3, 4), "a V-LoRA of 3 layers, rank 4")
    bad = [
        ("a partial layer", {"change": lambda lo: lo.pop()}, "7 a layer"),
        ("rank apart from its cfg", {"cfg": {"r": 3}}, "not of rank 3"),
        ("ranks apart", {"change": lambda lo: lo[3].update(A=np.zeros((3, q)), B=np.zeros((D, 3)))}, "not of rank 2"),
        ("an adapter of another input", {"change": lambda lo: lo[9].update(A=np.zeros((R, 7)))},
         "layer 1: adapters that do not fit"),
        ("o of another output", {"change": lambda lo: lo[3].update(B=np.zeros((9, R)))}, "layer 0: adapters"),
        ("down of another input", {"change": lambda lo: lo[6].update(A=np.zeros((R, 15)))}, "do not fit"),
        ("a third tensor", {"change": lambda lo: lo[0].update(C=np.zeros(2))}, "not A and B"),
        ("PEFT's lora_alpha", {"cfg": {"lora_alpha": 16}}, "names ['lora_alpha']"),
        ("an alpha", {"cfg": {"alpha": 2}}, "has no scale"),
    ]
    for what, kw, because in bad:
        check(refused(export_dv.check_vlora, ck(**kw), because), f"a V-LoRA checkpoint with {what}: refused")
    check(refused(export_dv.check_vlora, {"cfg": {"r": 2}}, "no adapters"), "a checkpoint without adapters: refused")


def dv_checkpoints():
    """export_dv.py's checks on bifill-dv's checkpoints (as loaded): only the plain residual canvas and the one-token
    bilinear selector."""
    def names(layers=2):
        n = {"norm", "mask_emb", "res_in.weight", "res_out.weight"}
        for layer in range(layers):
            n |= {f"layers.{layer}.ln1", f"layers.{layer}.ln2"}
            n |= {f"layers.{layer}.{x}.{m}" for x in ("q", "k", "v", "o", "gate", "up", "down") for m in ("A", "B")}
        return n
    plain = {"residual": True, "mixers": []}
    check(export_dv.canvas_layers(plain, names(3)) == 3, "a residual canvas of 3 layers")
    check(export_dv.canvas_layers({**plain, "soft_in": False, "see_guess": False, "res_k": 16}, names()) == 2,
          "flags off, z over 16: the plain residual canvas")
    for what, cfg, n, because in (
            ("mixers", {"residual": True, "mixers": ["m0"]}, names(), "mixers"),
            ("no residual context", {"residual": False}, names(), "without its residual context"),
            ("a spine", plain, names() | {"spine.w"}, "spine"),
            ("a see-gate", plain, names() | {"layers.0.see_gate"}, "see-gate"),
            ("a second residual", plain, names() | {"res_h.weight"}, "second residual"),
            ("a tensor missing", plain, names() - {"layers.1.up.B"}, "missing"),
            ("an unknown tensor", plain, names() | {"layers.0.q.C"}, "unknown"),
            # cfg flags that change the canvas pass with no tensor of their own (bifill-dv's evalgen.py reads them)
            ("soft inputs (soft_in)", {**plain, "soft_in": True}, names(), "soft_in"),
            ("the guesses seen (see_guess)", {**plain, "see_guess": True}, names(), "see_guess"),
            ("a see-gate (cfg)", {**plain, "see_guess": True, "see_gate": True}, names(), "see_gate"),
            ("a second residual (cfg)", {**plain, "residual_h": True}, names(), "residual_h"),
            ("a candidate memory", {**plain, "mem_layers": [3]}, names(), "mem_layers"),
            ("z over 8 candidates", {**plain, "res_k": 8}, names(), "sums 8 candidates")):
        check(refused(lambda c: export_dv.canvas_layers(c, n), cfg, because), f"a canvas with {what}: refused")
    sel = {"sel": {"A.weight": 0, "H.weight": 0, "B.weight": 0, "alpha": 0}, "r": 256}
    check(export_dv.check_selector(sel) == 256, "the one-token selector of rank 256")
    for what, ck, because in (("two-token", {**sel, "two": True}, "two-token and MLP"),
                              ("MLP", {**sel, "mlp": True}, "two-token and MLP"),
                              ("one-matrix", {"sel": {"A.weight": 0}, "r": 8}, "selector tensors")):
        check(refused(export_dv.check_selector, ck, because), f"a {what} selector: refused")


def dv_settings():
    """dv.json as export_dv.py writes it (dv_meta, no torch): the fill is the run's (--fill, default bifill-dv's TAUS
    0.9), never the checkpoint's training TAU, which goes to its source; causal and the ranks are the checkpoint's."""
    cfg = {"n": 32, "rank": 256, "residual": True, "res_k": 16, "canvas_causal": True, "tau": 0.5}
    meta = export_dv.dv_meta(cfg, 24, 128, source={"canvas": "c.pt"})
    check(meta["fill"] == 0.9 and meta["source"] == {"canvas": "c.pt", "tau": 0.5},
          f"fill 0.9 by default whatever the checkpoint's TAU (0.5, in source): {meta['fill']}, {meta['source']}")
    check(meta["causal"] is True and meta["slots"] == 32 and meta["rank"] == 256 and meta["selector"] == {"rank": 128}
          and meta["topk"] == 16 and meta["res_k"] == 16 and meta["tree"] == "beam:31:8:cov",
          f"the checkpoint's causal, slots and ranks; 16 candidates; DYN 31:8:cov: {meta}")
    meta = export_dv.dv_meta(cfg, 24, 128, slots=64, tree="pool:31:7:1", fill=0.7)
    check(meta["fill"] == 0.7 and meta["slots"] == 64 and meta["tree"] == "pool:31:7:1",
          "--fill 0.7 --slots 64 --tree pool:31:7:1")
    check(refused(lambda f: export_dv.dv_meta(cfg, 24, 128, fill=f), 1.5, "not a probability"), "--fill 1.5: refused")
    check(refused(lambda t: export_dv.dv_meta(cfg, 24, 128, slots=4, tree=t), "beam:31:2", "fewer than 31"),
          "a tree the slots cannot hold: refused")


def draft_taps():
    """A DFlash2 drafter's target_layer_ids as its taps (core/modelfile/drafter_dir.cpp reads a directory the same
    way): the layers whose outputs it reads, or, where the list reaches num_target_layers, the hidden states."""
    print("a DFlash2 drafter's config")
    drafter = {"architectures": ["DFlash2DraftModel"], "hidden_size": 64, "intermediate_size": 128,
               "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
               "vocab_size": 100, "is_causal": False}
    # z-lab's for Qwen3.8-27B: five of 64 layers.
    zlab = dict(drafter, num_target_layers=64,
                dflash_config={"block_size": 8, "mask_token_id": 3, "target_layer_ids": [5, 19, 33, 47, 61]})
    d = canonical_draft_config(zlab, None)
    check(d["taps"] == [6, 20, 34, 48, 62] and "softcap" not in d and "softcap_mode" not in d,
          "target_layer_ids below num_target_layers: the state after each layer, no soft cap")
    # Inco's for Inkling-2-Mini: four of 48 visits' hidden states, the last the final norm's output.
    inco = dict(drafter, num_target_layers=48,
                dflash_config={"block_size": 8, "mask_token_id": 3, "target_layer_ids": [5, 23, 41, 48],
                               "final_logit_softcapping": 30.0, "selector_softcap_mode": "after_correction"})
    d = canonical_draft_config(inco, None)
    check(d["taps"] == [5, 23, 41, 48] and d["softcap"] == 30.0 and d["softcap_mode"] == "after_correction",
          "target_layer_ids that reach num_target_layers: the hidden states as numbered, the soft cap and its mode")
    check(refused(lambda c: canonical_draft_config(c, None),
                  dict(inco, dflash_config=dict(inco["dflash_config"], selector_softcap_mode="sometimes")),
                  "selector_softcap_mode"), "a soft cap mode the engine does not run is refused")
    check(refused(lambda c: canonical_draft_config(c, None),
                  dict(inco, dflash_config=dict(inco["dflash_config"], final_logit_softcapping=-1.0)),
                  "final_logit_softcapping"), "a soft cap below zero is refused")


def eagle4_config():
    """EAGLE-Corp/Qwen3.8-27B-EAGLE4's config.json as published (core/modelfile/drafter_dir.cpp reads it the same way):
    its taps the outputs of target layers 3, 31 and 59, a window of 4096, 128 candidates of a rank-128 head."""
    print("an EAGLE-4 drafter's config")
    published = {
        "architectures": ["LlamaForCausalLMEagle4"], "attention_bias": False, "attention_dropout": 0.0,
        "bos_token_id": 248044, "draft_vocab_size": 248320, "dtype": "bfloat16",
        "eagle_config": {"eagle_aux_hidden_state_layer_ids": [3, 31, 59], "use_aux_hidden_state": True},
        "efficient_lm_head_config": {"num_candidates": 128, "rank": 128, "type": "low_rank"},
        "eos_token_id": 248044, "fc_norm": True, "head_dim": 128, "hidden_act": "silu", "hidden_size": 5120,
        "initializer_range": 0.02, "intermediate_size": 16384, "layer_types": ["sliding_attention"],
        "max_position_embeddings": 262144, "mlp_bias": False, "model_type": "llama", "norm_output": True,
        "num_attention_heads": 40, "num_aux_hidden_states": 3, "num_hidden_layers": 1, "num_key_value_heads": 40,
        "pad_token_id": 248044, "pretraining_tp": 1, "rms_norm_eps": 1e-06,
        "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"}, "sliding_window": 4096,
        "target_hidden_size": 5120, "tie_word_embeddings": False, "transformers_version": "5.8.1",
        "use_cache": True, "use_sliding_window": True, "vocab_size": 248320, "torch_dtype": "bfloat16"}
    d = canonical_draft_config(published, {"hidden": 5120, "vocab": 248320, "layers": 64})
    check(d == {"kind": "eagle4", "hidden": 5120, "layers": 1, "n_heads": 40, "n_kv": 40, "head_dim": 128,
                "inter": 16384, "eps": 1e-06, "rope_theta": 10000000.0, "window": 4096, "taps": [4, 32, 60],
                "cand_rank": 128, "n_cands": 128, "target": {"hidden": 5120, "vocab": 248320, "layers": None}},
          f"the released drafter's canonical config: {d}")
    check(refused(lambda c: canonical_draft_config(c, {"hidden": 5120, "vocab": 248320, "layers": 32}), published,
                  "outside the target's layers"), "taps past the target's layers: refused")
    check(refused(lambda c: canonical_draft_config(c, None), dict(published, norm_output=False), "norm_output"),
          "a drafter that carries its unnormalised output on: refused")
    # What the engine refuses to open (models/drafters/eagle4.cpp) is refused before a tensor is read: more taps than
    # FLYX_EAGLE4_MAX_TAPS, a rotary base or a norm epsilon not finite and above zero.
    def taps(n):
        return dict(published, num_aux_hidden_states=n,
                    eagle_config=dict(published["eagle_config"], eagle_aux_hidden_state_layer_ids=list(range(n))))
    with open(os.path.join(HERE, "..", "..", "..", "include", "flyx", "backend_abi.h"), encoding="utf-8") as f:
        abi = re.search(r"#define FLYX_EAGLE4_MAX_TAPS (\d+)", f.read())
    check(abi and int(abi.group(1)) == eagle4.MAX_TAPS == 8, "the converter's tap limit is the ABI's")
    d = canonical_draft_config(taps(8), {"hidden": 5120, "vocab": 248320, "layers": 64})
    check(d["taps"] == list(range(1, 9)), f"eight taps, the most the engine takes: {d['taps']}")
    check(refused(lambda c: canonical_draft_config(c, None), taps(9), "9 taps (the engine takes 1..8)"),
          "nine taps: refused")
    for theta in (0, 0.0, -1.0, float("inf"), float("nan"), True):
        check(refused(lambda c: canonical_draft_config(c, None),
                      dict(published, rope_parameters={"rope_theta": theta, "rope_type": "default"}), "rope_parameters"),
              f"rope_theta {theta!r}: refused")
    for eps in (0, -1e-6, float("inf"), float("nan")):
        check(refused(lambda c: canonical_draft_config(c, None), dict(published, rms_norm_eps=eps), "rms_norm_eps"),
              f"rms_norm_eps {eps!r}: refused")


def main():
    rotary()
    draft_taps()
    eagle4_config()
    with tempfile.TemporaryDirectory() as tmp:
        post_processor(tmp)
    with tempfile.TemporaryDirectory() as tmp:
        dv_drafter(tmp)
    with tempfile.TemporaryDirectory() as tmp:
        vlora_files(tmp)
    with tempfile.TemporaryDirectory() as tmp:
        file_format(tmp)
    dv_checkpoints()
    vlora_checkpoints()
    dv_settings()
    sys.exit(finish("convert_test"))


if __name__ == "__main__":
    main()
