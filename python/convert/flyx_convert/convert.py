# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""HuggingFace Qwen3.5 (+ optional DFlash2 or DSpine drafter), Qwen3 or Llama (+ optional DV) -> one .flyx file.

    python -m flyx_convert --model Qwen3.5-4B/ --draft Qwen3.5-4B-DFlash2/ -o qwen35-4b.flyx
    python -m flyx_convert --draft Qwen3.8-27B-DFlash2/ -o qwen38-27b-dflash2.flyx   # a drafter file
    python -m flyx_convert --draft Qwen3.8-27B-DSpine-b16/ -o qwen38-27b-dspine.flyx  # a DSpine drafter file
    python -m flyx_convert --draft Qwen3.8-27B-EAGLE4/ -o qwen38-27b-eagle4.flyx       # an EAGLE-4 drafter file
    python -m flyx_convert --model MiniCPM5-1B/ -o minicpm5-1b.flyx                     # Llama, no drafter
    python -m flyx_convert --model MiniCPM5-1B/ --dv minicpm5-dv/ -o minicpm5-1b-dv-res2.flyx  # Llama with DV
    python -m flyx_convert --model MiniCPM5-1B/ --lora minicpm5-vlora/ -o minicpm5-1b-vlora.flyx  # its V-LoRA

A drafter file holds a drafter (DFlash2, DSpine: dspine.py, or EAGLE-4: eagle4.py) alone: the engine takes
it beside a model of any format, a GGUF included (--drafter-file).

A DV drafter (bifill-dv's canvas and selector, docs/design/dv.md) comes from
the directory python/convert/export_dv.py writes, for a Llama (dv.py); the
target's own adapters (a V-LoRA, lora.py) from export_dv.py --vlora's, for a
Llama or a Qwen3: with --dv, the V-LoRA of the DV's directory when it has one
(the canvas and selector were made for that target), with --lora one alone.

Qwen3 (dense) and Llama (MiniCPM5: Llama's layers without q/k norms) convert
the same way; Qwen3's Qwen2 pre-tokenizer split, Qwen3.5's and
MiniCPM5's (numbers in groups of three first) are supported
(tokenizer_export.py).

Weights are copied as stored (BF16 stays BF16).  Tensor names become the
engine's canonical names; the only data change is a row reorder of the
Qwen3.5 attention query projection so that all query heads come before all
gate heads.  Unknown tensors are an error, not a silent skip.  The model's
own multi-token-prediction layer (``mtp.*``), when present, is kept as a
second drafter (``mtp.`` names).  The header's ``arch`` names the model
definition the engine reads it with (models/<arch>).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import struct
import sys

import numpy as np

from .dspine import canonical_dspine_config, check_dspine, dspine_tensors
from .dv import add_dv
from .eagle4 import canonical_eagle4_config, check_eagle4, eagle4_tensors
from .fileformat import NUMPY, Writer
from .lora import add_lora, has_lora
from .rules import ATTN_LAYER_KEYS, DRAFT_LAYER_RULES, DRAFT_SKIP, check_tensors, layer_fields, map_name
from .tokenizer_export import export as export_tokenizer

ST_DTYPES = {"BF16": "bf16", "F16": "f16", "F32": "f32", "F64": "f64", "I32": "i32", "I64": "i64", "U8": "u8",
             "F8_E4M3": "f8e4m3"}


class SafeTensors:
    """Minimal safetensors reader (raw bytes; no dtype conversion)."""

    def __init__(self, directory: str):
        self.files = sorted(glob.glob(os.path.join(glob.escape(directory), "*.safetensors")))
        if not self.files:
            raise FileNotFoundError(f"no *.safetensors in {directory}")
        self.index = {}
        self._handles: dict[str, object] = {}
        for path in self.files:
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            for name, rec in hdr.items():
                if name != "__metadata__":
                    self.index[name] = (path, 8 + n + rec["data_offsets"][0],
                                        rec["data_offsets"][1] - rec["data_offsets"][0],
                                        ST_DTYPES[rec["dtype"]], rec["shape"])

    def names(self):
        return list(self.index)

    def read(self, name: str):
        path, off, size, dtype, shape = self.index[name]
        f = self._handles.get(path)
        if f is None:
            f = open(path, "rb")
            self._handles[path] = f
        f.seek(off)
        buf = f.read(size)
        return np.frombuffer(buf, dtype=NUMPY[dtype]).reshape(shape), dtype

    def close(self):
        for f in self._handles.values():
            f.close()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


def _text_config(cfg: dict) -> dict:
    return cfg.get("text_config", cfg)


def _plain_rope(tc: dict) -> dict:
    """The rotary's parameters, refused unless plain: the engine turns position p by p theta^(-2i/d), unscaled.
    transformers reads the scaling's kind from `rope_type` or from the older `type` ({"type": "linear", "factor":
    4} in many Llama-family configs), in `rope_parameters` or the older `rope_scaling`: each is checked, and a
    factor other than 1 is refused whatever its kind is called."""
    given = [rp for rp in (tc.get("rope_parameters"), tc.get("rope_scaling")) if rp]
    for rp in given:
        kind = rp.get("rope_type") or rp.get("type") or "default"
        if kind != "default":
            raise ValueError(f"unsupported rope type {kind}")
        if rp.get("factor") not in (None, 1, 1.0):
            raise ValueError(f"scaled rotary (factor {rp['factor']}) is not supported")
    return given[0] if given else {}


def _common_transformer_config(tc: dict, *, default_tie: bool, default_max_pos: int) -> tuple[dict, dict, int]:
    rp = _plain_rope(tc)
    hd = tc.get("head_dim") or tc["hidden_size"] // tc["num_attention_heads"]
    if tc.get("attention_bias"):
        raise ValueError("attention bias is not supported")
    base = {
        "hidden": tc["hidden_size"], "layers": tc["num_hidden_layers"],
        "vocab": tc["vocab_size"], "inter": tc["intermediate_size"],
        "n_heads": tc["num_attention_heads"], "n_kv": tc["num_key_value_heads"], "head_dim": hd,
        "rope_theta": float(rp.get("rope_theta", tc.get("rope_theta", 10000.0))),
        "eps": tc["rms_norm_eps"],
        "tie_embed": bool(tc.get("tie_word_embeddings", default_tie)),
        "max_pos": tc.get("max_position_embeddings", default_max_pos),
    }
    return base, rp, hd


def canonical_target_config(tc: dict) -> dict:
    base, rp, hd = _common_transformer_config(tc, default_tie=True, default_max_pos=262144)
    if not tc.get("attn_output_gate", True):
        raise ValueError("models without the attention output gate are not supported")
    lt = tc.get("layer_types") or [
        "full_attention" if (i + 1) % tc.get("full_attention_interval", 4) == 0 else "linear_attention"
        for i in range(tc["num_hidden_layers"])]
    return {
        **base,
        "layer_types": ["attn" if t == "full_attention" else "gdn" for t in lt],
        "rot_dim": int(hd * rp.get("partial_rotary_factor", tc.get("partial_rotary_factor", 1.0))),
        "lin_nk": tc["linear_num_key_heads"], "lin_nv": tc["linear_num_value_heads"],
        "lin_dk": tc["linear_key_head_dim"], "lin_dv": tc["linear_value_head_dim"],
        "conv_k": tc["linear_conv_kernel_dim"],
    }


# What some Llama-family checkpoints scale beyond Llama's math (MiniCPM 1 to 4's muP, Granite's multipliers): the
# engine's Llama has none of it.
_LLAMA_SCALES = ("scale_emb", "scale_depth", "dim_model_base", "embedding_multiplier", "residual_multiplier",
                 "logits_scaling", "attention_multiplier")


def canonical_dense_config(tc: dict) -> dict:
    """Qwen3 dense and Llama: every layer attention (no output gate), plain RMSNorm weights; Qwen3 normalises each
    head's q and k before the rotary, Llama does not (models/llama)."""
    base, _, hd = _common_transformer_config(tc, default_tie=False, default_max_pos=40960)
    if tc.get("mlp_bias"):
        raise ValueError("MLP bias is not supported")
    if tc.get("use_sliding_window"):
        raise ValueError("sliding-window attention is not supported")
    if tc.get("hidden_act", "silu") != "silu":
        raise ValueError(f"activation {tc.get('hidden_act')!r} is not supported (silu)")
    scaled = [k for k in _LLAMA_SCALES if tc.get(k) not in (None, 1, 1.0)]
    if scaled:
        raise ValueError(f"scaled embeddings, residuals or logits are not supported ({', '.join(scaled)})")
    return {
        **base,
        "layer_types": ["attn"] * tc["num_hidden_layers"],
        "rot_dim": hd,
    }


def canonical_gemma4_config(tc: dict) -> dict:
    """Gemma 4's dense layers (models/gemma4): sliding and global layers, each kind its head width, K/V heads,
    rotary and theta; v from k's projection on the global ones (attention_k_eq_v); GELU (tanh) MLPs; the final logits
    softcapped; experts beside every layer's MLP (26B-A4B: enable_moe_block).  Per-layer embeddings (E2B, E4B) and
    shared K/V layers are refused for now."""
    for key, what in (("hidden_size_per_layer_input", "per-layer embeddings (E2B, E4B)"),
                      ("num_kv_shared_layers", "shared K/V layers"),
                      ("use_double_wide_mlp", "a double-wide MLP"), ("attention_bias", "attention biases")):
        if tc.get(key):
            raise ValueError(f"Gemma 4 with {what} is not supported yet")
    if tc.get("hidden_activation", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
        raise ValueError(f"activation {tc.get('hidden_activation')!r} is not supported (gelu_pytorch_tanh)")
    rp = tc.get("rope_parameters") or {}
    full, swa = rp.get("full_attention") or {}, rp.get("sliding_attention") or {}
    if full.get("rope_type", "proportional") != "proportional" or swa.get("rope_type", "default") != "default" or \
            swa.get("partial_rotary_factor", 1.0) != 1.0 or full.get("factor", 1.0) != 1.0:
        raise ValueError(f"unsupported Gemma 4 rotary: {rp}")
    lt = tc["layer_types"]
    if len(lt) != tc["num_hidden_layers"] or any(t not in ("sliding_attention", "full_attention") for t in lt):
        raise ValueError("Gemma 4 layer_types: sliding_attention or full_attention, one a layer")
    k_eq_v = bool(tc.get("attention_k_eq_v", False))
    # The global layers' heads: global_head_dim and num_global_key_value_heads (as Google's configs give them), or a
    # global layer's own entries in per_layer_config (as transformers writes them since)
    plc = tc.get("per_layer_config") or {}
    full_own = next((plc.get(str(i)) or {} for i, t in enumerate(lt) if t == "full_attention"), {})
    hd = tc["head_dim"]
    hd_full = full_own.get("head_dim") or tc.get("global_head_dim") or hd
    n_kv_full = full_own.get("num_key_value_heads") or (tc.get("num_global_key_value_heads") if k_eq_v else None)
    return {
        "hidden": tc["hidden_size"], "layers": tc["num_hidden_layers"], "vocab": tc["vocab_size"],
        "inter": tc["intermediate_size"], "n_heads": tc["num_attention_heads"],
        "n_kv": tc["num_key_value_heads"], "head_dim": hd,
        "n_kv_full": n_kv_full or tc["num_key_value_heads"], "head_dim_full": hd_full,
        "layer_types": ["sliding" if t == "sliding_attention" else "full" for t in lt],
        "window": tc["sliding_window"],
        "rope_theta": float(full.get("rope_theta", 1000000.0)), "rope_theta_swa": float(swa.get("rope_theta", 10000.0)),
        "rot_dim_full": 2 * int(full.get("partial_rotary_factor", 0.25) * hd_full // 2),
        "k_eq_v": k_eq_v, "eps": tc["rms_norm_eps"], "softcap": float(tc.get("final_logit_softcapping") or 0.0),
        "tie_embed": bool(tc.get("tie_word_embeddings", True)), "max_pos": tc.get("max_position_embeddings", 131072),
        **({"experts": tc["num_experts"], "top_k": tc["top_k_experts"], "moe_inter": tc["moe_intermediate_size"]}
           if tc.get("enable_moe_block") else {}),
    }


def arch_of(hf_cfg: dict) -> str:
    t = _text_config(hf_cfg).get("model_type") or hf_cfg.get("model_type", "")
    if t in ("gemma4_text", "gemma4", "gemma4_unified_text", "gemma4_unified"):
        return "gemma4"
    if t == "qwen3":
        return "qwen3"
    if t in ("llama", "minicpm5"):  # MiniCPM5 publishes LlamaForCausalLM checkpoints
        return "llama"
    if t.startswith("qwen3_5") or t.startswith("qwen3_next"):
        return "qwen35"
    raise ValueError(f"model type {t!r} is not supported (qwen3_5, qwen3, llama, gemma4)")


# The head's candidates DSpark's Markov walk takes a step (its greedy walk is over the whole vocabulary): a wide pick
# of the 1-bit draft head's (kWalkMax).  Over 12 prompts each of gsm8k, humaneval, mt-bench and alpaca, Qwen3-8B with
# DeepSpec's dspark_qwen3_8b_block7 in its own evaluator accepted 6.58 / 5.89 / 3.70 / 3.82 tokens a round over the
# whole vocabulary, 6.58 / 5.86 / 3.68 / 3.78 over the head's top 1024, 6.57 / 5.82 / 3.62 / 3.73 over its top 256,
# 6.39 / 5.70 / 3.53 / 3.65 over its top 64 and 6.10 / 5.29 / 3.32 / 3.45 over its top 16.
DSPARK_CANDIDATES = 1024


# The block drafters a config.json names by its architecture, and the kind each converts to (core/modelfile/
# drafter_dir.cpp's kDrafterArchitectures).
_DRAFTER_ARCHITECTURES = {"DFlash2DraftModel": "dflash2", "DFlashDraftModel": "dflash", "Qwen3DSparkModel": "dspark",
                          "DSpineDraftModel": "dspine", "LlamaForCausalLMEagle4": "eagle4"}


def _drafter_config_kind(sc: dict) -> str:
    """The block drafter a config.json is: by its architecture, else speculators' type ("dflash2"); "" for none."""
    for a in sc.get("architectures", []):
        if a in _DRAFTER_ARCHITECTURES:
            return _DRAFTER_ARCHITECTURES[a]
    return "dflash2" if sc.get("speculators_model_type") == "dflash2" else ""


def canonical_dspark_config(sc: dict, target: dict | None) -> dict:
    """A DSpark drafter's config (DeepSpec's "Qwen3DSparkModel"): DFlash2's block drafter without the convolutions
    (conv_taps 0), every row of the block a draft (draft_from 0: row k predicts the token after the anchor's k-th
    successor), and a Markov head (markov_w1, markov_w2) in the selector's place with no hidden gate (sel_gate false).
    Its embedding and head are frozen copies of the target's (the engine uses the target's); its confidence head is
    left out (the rounds do not use it)."""
    rank = int(sc["markov_rank"])
    if rank <= 0 or str(sc["markov_head_type"]).lower() != "vanilla":
        raise ValueError("DSpark: a vanilla Markov head (markov_rank > 0) is supported")
    ids = [int(i) for i in sc["target_layer_ids"]]
    if any(i < 0 for i in ids):
        raise ValueError("DSpark: a tap of the embedding (target layer -1) is not supported")
    taps = [i + 1 for i in ids]
    types = sc.get("layer_types") or ["full_attention"] * sc["num_hidden_layers"]
    if any(t != "full_attention" for t in types):
        raise ValueError(f"DSpark layer types {sorted(set(types))}: full attention is supported")
    if target is not None:
        if sc["vocab_size"] != target["vocab"] or sc["hidden_size"] != target["hidden"]:
            raise ValueError("drafter vocabulary or width differs from the target's")
        if any(not 1 <= t <= target["layers"] for t in taps):
            raise ValueError(f"drafter taps {taps} outside the target's {target['layers']} layers")
    return {
        "kind": "dspark", **layer_fields(sc),
        "block": sc["block_size"], "mask_token": sc["mask_token_id"], "taps": taps,
        "conv_taps": 0, "conv_group": 1, "sel_rank": rank, "sel_topk": min(DSPARK_CANDIDATES, sc["vocab_size"]),
        "window": 0, "draft_from": 0, "sel_gate": False,
        "target": {"hidden": sc["hidden_size"], "vocab": sc["vocab_size"],
                   "layers": target["layers"] if target else sc.get("num_target_layers")},
    }


def canonical_draft_config(sc: dict, target: dict | None) -> dict:
    """A block drafter's config: a DSpark drafter's (canonical_dspark_config), a DSpine drafter's (dspine.py), or a
    DFlash2 drafter's, from either way it is published:
      speculators' ("speculators_model_type": "dflash2"; the layers under "transformer_layer_config";
        aux_hidden_state_layer_ids: indices into the target's hidden states, 0 the embedding);
      z-lab's and Inco's ("architectures": ["DFlash2DraftModel"], "dflash_config"; target_layer_ids: the target
        layers whose outputs it reads, hidden_states[id + 1] in their code; a list that reaches num_target_layers
        counts the hidden states themselves instead, the last the final norm's output, as Inco's drafter for
        Inkling-2 does: its reference hands the ids to the capture DSpark's go to, as they are);
    or a DFlash drafter's (z-lab's "DFlashDraftModel", what DFlash2 grew from; block_size beside its dflash_config):
    DFlash2's block without convolutions (conv_taps 0) or selector (sel_rank 0, sel_topk 1: each row's draft the
    head's best).
    taps: the target's hidden state after that many layers; after all of them, the final norm's output.  softcap,
    softcap_mode: the selector's soft cap on a candidate's logit (final_logit_softcapping), before it adds the
    pair's correction or after (selector_softcap_mode).  window: each position attends to the positions less than
    that far from it, both ways (0: all).  target: the model's canonical config, or None for a drafter file of its
    own (the engine checks it against the model it is given with; "target" below names what it was made for)."""
    kind = _drafter_config_kind(sc)
    if kind == "dspark":
        return canonical_dspark_config(sc, target)
    if kind == "dspine":
        return canonical_dspine_config(sc, target)
    if kind == "eagle4":
        return canonical_eagle4_config(sc, target)
    dflash = kind == "dflash"
    if "dflash_config" in sc:
        if kind not in ("dflash2", "dflash"):
            raise ValueError(f"drafter architectures {sc.get('architectures')!r}: no block drafter flyx reads")
        dc, tl = sc["dflash_config"], sc
        for key in ("conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k"):
            if dflash and key in dc:
                raise ValueError(f"DFlash drafter: DFlash2's {key}")
        ids = [int(i) for i in dc["target_layer_ids"]]
        n_target = sc.get("num_target_layers")
        hidden_states = isinstance(n_target, int) and not isinstance(n_target, bool) and n_target in ids
        taps = ids if hidden_states else [i + 1 for i in ids]
        vocab, target_hidden = tl["vocab_size"], tl["hidden_size"]
        causal = sc.get("is_causal")  # their attention: None = causal where the layer slides
    else:
        if sc.get("speculators_model_type") != "dflash2":
            raise ValueError(f"drafter type {sc.get('speculators_model_type')!r} is not supported (need dflash2)")
        if sc.get("sample_from_anchor", False):
            raise ValueError("sample_from_anchor=True is not supported")
        dc, tl = sc, sc["transformer_layer_config"]
        taps = [int(i) for i in sc["aux_hidden_state_layer_ids"]]
        vocab = sc.get("draft_vocab_size", tl["vocab_size"])
        target_hidden = sc.get("target_hidden_size") or tl["hidden_size"]
        causal = False
    types = tl.get("layer_types") or ["full_attention"] * tl["num_hidden_layers"]
    if all(t == "full_attention" for t in types):
        window = 0
    elif all(t == "sliding_attention" for t in types) and tl.get("sliding_window"):
        window = int(tl["sliding_window"])
    else:
        raise ValueError(f"draft layer types {sorted(set(types))}: all full or all sliding with one window")
    if causal is None:
        causal = window > 0
    if causal:
        raise ValueError("causal DFlash2 layers are not supported (the block attends both ways)")
    if target is not None:
        if vocab != target["vocab"]:
            raise ValueError("drafter vocabulary differs from the target's")
        if target_hidden != target["hidden"]:
            raise ValueError("drafter target_hidden_size does not match the model")
        if any(not 1 <= t <= target["layers"] for t in taps):
            raise ValueError(f"drafter taps {taps} outside the target's {target['layers']} layers")
    draft = {
        "kind": kind, **layer_fields(tl),
        "block": sc["block_size"] if dflash else dc["block_size"], "mask_token": dc["mask_token_id"], "taps": taps,
        "conv_taps": 0 if dflash else dc.get("conv_kernel_size", 2),
        "conv_group": 1 if dflash else dc.get("conv_group_size", 16),
        "sel_rank": 0 if dflash else dc.get("selector_rank", 256),
        "sel_topk": 1 if dflash else dc.get("selector_top_k", 16),
        "window": window, "draft_from": 1, "sel_gate": not dflash,
        "target": {"hidden": target_hidden, "vocab": vocab,
                   "layers": target["layers"] if target else sc.get("num_target_layers")},
    }
    softcap = dc.get("final_logit_softcapping")
    if isinstance(softcap, (int, float)) and not isinstance(softcap, bool):
        if not math.isfinite(softcap) or softcap < 0:
            raise ValueError("draft final_logit_softcapping out of range")
        if softcap > 0:
            draft["softcap"] = float(softcap)
    mode = dc.get("selector_softcap_mode")
    if isinstance(mode, str):
        if mode not in ("before_correction", "after_correction"):
            raise ValueError(f"unsupported draft selector_softcap_mode: {mode}")
        draft["softcap_mode"] = mode
    return draft


_T_RULES = [
    (r"embed_tokens\.weight", "embed"),
    (r"norm\.weight", "norm"),
    (r"layers\.(\d+)\.input_layernorm\.weight", "L{0}.in_norm"),
    (r"layers\.(\d+)\.post_attention_layernorm\.weight", "L{0}.post_norm"),
    (r"layers\.(\d+)\.linear_attn\.in_proj_qkv\.weight", "L{0}.gdn.qkv"),
    (r"layers\.(\d+)\.linear_attn\.in_proj_z\.weight", "L{0}.gdn.z"),
    (r"layers\.(\d+)\.linear_attn\.in_proj_b\.weight", "L{0}.gdn.b"),
    (r"layers\.(\d+)\.linear_attn\.in_proj_a\.weight", "L{0}.gdn.a"),
    (r"layers\.(\d+)\.linear_attn\.conv1d\.weight", "L{0}.gdn.conv"),
    (r"layers\.(\d+)\.linear_attn\.A_log", "L{0}.gdn.A_log"),
    (r"layers\.(\d+)\.linear_attn\.dt_bias", "L{0}.gdn.dt_bias"),
    (r"layers\.(\d+)\.linear_attn\.norm\.weight", "L{0}.gdn.norm"),
    (r"layers\.(\d+)\.linear_attn\.out_proj\.weight", "L{0}.gdn.out"),
    (r"layers\.(\d+)\.self_attn\.q_proj\.weight", "L{0}.attn.q"),
    (r"layers\.(\d+)\.self_attn\.k_proj\.weight", "L{0}.attn.k"),
    (r"layers\.(\d+)\.self_attn\.v_proj\.weight", "L{0}.attn.v"),
    (r"layers\.(\d+)\.self_attn\.o_proj\.weight", "L{0}.attn.o"),
    (r"layers\.(\d+)\.self_attn\.q_norm\.weight", "L{0}.attn.q_norm"),
    (r"layers\.(\d+)\.self_attn\.k_norm\.weight", "L{0}.attn.k_norm"),
    (r"layers\.(\d+)\.mlp\.gate_proj\.weight", "L{0}.mlp.gate"),
    (r"layers\.(\d+)\.mlp\.up_proj\.weight", "L{0}.mlp.up"),
    (r"layers\.(\d+)\.mlp\.down_proj\.weight", "L{0}.mlp.down"),
]
# Gemma 4's layer: a norm on either side of attention and of the MLP (its post_attention_layernorm is attention's
# own, after it; pre_feedforward_layernorm the MLP's input norm, Qwen's post_attention_layernorm), the layer's scalar.
_G4_RULES = [
    (r"embed_tokens\.weight", "embed"),
    (r"norm\.weight", "norm"),
    (r"layers\.(\d+)\.input_layernorm\.weight", "L{0}.in_norm"),
    (r"layers\.(\d+)\.post_attention_layernorm\.weight", "L{0}.attn.post_norm"),
    (r"layers\.(\d+)\.pre_feedforward_layernorm\.weight", "L{0}.post_norm"),
    (r"layers\.(\d+)\.post_feedforward_layernorm\.weight", "L{0}.mlp.post_norm"),
    (r"layers\.(\d+)\.layer_scalar", "L{0}.layer_scalar"),
    (r"layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight", "L{0}.attn.{1}"),
    (r"layers\.(\d+)\.self_attn\.(q|k)_norm\.weight", "L{0}.attn.{1}_norm"),
    (r"layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight", "L{0}.mlp.{1}"),
    # a layer's experts beside its MLP (26B-A4B)
    (r"layers\.(\d+)\.post_feedforward_layernorm_1\.weight", "L{0}.mlp.post_norm_1"),
    (r"layers\.(\d+)\.pre_feedforward_layernorm_2\.weight", "L{0}.moe.norm"),
    (r"layers\.(\d+)\.post_feedforward_layernorm_2\.weight", "L{0}.moe.post_norm"),
    (r"layers\.(\d+)\.router\.proj\.weight", "L{0}.moe.router"),
    (r"layers\.(\d+)\.router\.scale", "L{0}.moe.router_in"),
    (r"layers\.(\d+)\.router\.per_expert_scale", "L{0}.moe.expert_scale"),
    (r"layers\.(\d+)\.experts\.gate_up_proj", "L{0}.moe.gate_up"),
    (r"layers\.(\d+)\.experts\.down_proj", "L{0}.moe.down"),
]
_G4_MOE_KEYS = ("mlp.post_norm_1", "moe.norm", "moe.post_norm", "moe.router", "moe.router_in", "moe.expert_scale",
                "moe.gate_up", "moe.down")
_G4_LAYER_KEYS = ("in_norm", "attn.post_norm", "post_norm", "mlp.post_norm", "layer_scalar", "attn.q", "attn.k",
                  "attn.o", "attn.q_norm", "attn.k_norm", "mlp.gate", "mlp.up", "mlp.down")
_D_RULES = DRAFT_LAYER_RULES + [
    (r"layers\.(\d+)\.attention_conv\.base_kernel", "L{0}.attn_conv.base"),
    (r"layers\.(\d+)\.attention_conv\.kernel_projection\.weight", "L{0}.attn_conv.proj"),
    (r"layers\.(\d+)\.mlp_conv\.base_kernel", "L{0}.mlp_conv.base"),
    (r"layers\.(\d+)\.mlp_conv\.kernel_projection\.weight", "L{0}.mlp_conv.proj"),
    (r"candidate_selector\.predecessor_codebook", "sel.pred"),
    (r"candidate_selector\.successor_codebook", "sel.succ"),
    (r"candidate_selector\.hidden_projection\.weight", "sel.hid"),
    (r"markov_head\.markov_w1\.weight", "sel.pred"),    # DSpark: W1[previous token]
    (r"markov_head\.markov_w2\.weight", "sel.succ"),    # W2: [vocab, rank]
]
_MTP_RULES = [
    (r"fc\.weight", "fc"),
    (r"pre_fc_norm_embedding\.weight", "norm_emb"),
    (r"pre_fc_norm_hidden\.weight", "norm_hidden"),
    (r"norm\.weight", "norm"),
] + [(pat, out) for pat, out in _T_RULES if pat.startswith("layers")]
# Present in some checkpoints, intentionally not used by the engine.
_T_SKIP = re.compile(r"^(model\.visual\.|visual\.|model\.vision_tower\.|model\.audio_tower\.|model\.embed_vision\."
                     r"|model\.embed_audio\.)")
_MTP_PREFIX = re.compile(r"^(model\.)?mtp\.")


def _read_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_optional_config(path: str) -> dict:
    return _read_config(path) if os.path.exists(path) else {}


def _quant_config(model_dir: str, hf_cfg: dict) -> dict | None:
    """NVFP4 checkpoints from NVIDIA ModelOpt (hf_quant_config.json or quantization_config), or compressed-tensors'
    nvfp4-pack-quantized (QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4): every quantized linear NVFP4, groups of 16."""
    q = hf_cfg.get("quantization_config")
    p = os.path.join(model_dir, "hf_quant_config.json")
    if q is None and os.path.exists(p):
        q = _read_config(p).get("quantization")
    if not q:
        return None
    if (q.get("quant_method") or "").lower() == "compressed-tensors":
        groups = list((q.get("config_groups") or {}).values())
        wq = [g.get("weights") or {} for g in groups]
        if q.get("format") != "nvfp4-pack-quantized" or not groups or any(
                x.get("num_bits") != 4 or x.get("type") != "float" or x.get("group_size") != 16 for x in wq):
            raise ValueError(f"compressed-tensors format {q.get('format')!r} is not supported (nvfp4-pack-quantized is)")
        # inputs in 4 bits as well (input_global_scale): calibrated for NVFP4 inputs, as ModelOpt's below
        a4 = all((g.get("input_activations") or {}).get("num_bits") == 4 for g in groups)
        return {"format": "nvfp4", "activations": "nvfp4" if a4 else "bf16", "producer": "compressed-tensors"}
    algo = (q.get("quant_algo") or "").upper()
    if algo != "NVFP4":
        raise ValueError(f"quantization {algo or q.get('quant_method')!r} is not supported (NVFP4 from ModelOpt is)")
    # ModelOpt's NVFP4 is W4A4: the checkpoint is calibrated for NVFP4 inputs (input_scale), and the engine gives
    # its prefill passes those.
    return {"format": "nvfp4", "activations": "nvfp4", "producer": (q.get("producer") or {}).get("name", "modelopt")}


# ModelOpt NVFP4 linear: NAME.weight (u8, two e2m1 per byte), NAME.weight_scale
# (e4m3, one per 16), NAME.weight_scale_2 (f32 global), NAME.input_scale (the
# calibrated scale of the FP4 inputs; this engine takes a scale per row instead).
# compressed-tensors' the same as NAME.weight_packed, NAME.weight_scale, NAME.weight_global_scale (1 / ModelOpt's
# weight_scale_2) and NAME.input_global_scale.
_QUANT_SUFFIX = re.compile(r"^(.*)\.(weight_scale_2|weight_scale|input_scale|weight_global_scale|input_global_scale)$")
_PACKED = ".weight_packed"


def _strip_target_prefix(name: str) -> str | None:
    for p in ("model.language_model.", "language_model.model.", "model."):
        if name.startswith(p):
            return name[len(p):]
    return None


def _reorder_q(w: np.ndarray, n_heads: int, head_dim: int) -> np.ndarray:
    """HF rows are [h0: q(d), gate(d) | h1: ...]; engine wants [all q | all gates]."""
    w = w.reshape(n_heads, 2, head_dim, -1)
    return np.ascontiguousarray(w.transpose(1, 0, 2, 3)).reshape(2 * n_heads * head_dim, -1)


def _base_models(model_dir: str) -> list[str]:
    """The base models a model directory's card names ("org/name": README.md's front matter, base_model).  flyx offers
    the drafters made for a model's own repository and, inferred, those of its base models (docs/models.md)."""
    out = []
    card = os.path.join(model_dir, "README.md")
    text = open(card, encoding="utf-8", errors="replace").read() if os.path.exists(card) else ""
    end = text.find("\n---", 3) if text.startswith("---\n") else -1
    lines = text[4:end].split("\n") if end > 0 else []
    for i, line in enumerate(lines):
        if not line.startswith("base_model:"):
            continue
        value = line[len("base_model:"):].strip()
        names = [value] if value else []
        for item in lines[i + 1:] if not value else []:  # a list: its items, up to the next key
            if not item.strip().startswith("- "):
                break
            names.append(item.strip()[2:].strip())
        out += [n.strip("'\"") for n in names if re.fullmatch(r"['\"]?[\w.-]+/[\w.-]+['\"]?", n)]
    return list(dict.fromkeys(out))


def convert(model_dir: str, out: str, draft_dir: str | None = None, log=print, activations: str | None = None,
            dv_dir: str | None = None, lora_dir: str | None = None, repo: str | None = None) -> None:
    """activations: what the file says its NVFP4 weights' inputs are in prefill passes ("nvfp4", "bf16"); None: what
    the checkpoint is calibrated for (ModelOpt's NVFP4: nvfp4).  dv_dir: a DV drafter (export_dv.py's directory),
    with the V-LoRA it holds; lora_dir: a V-LoRA alone (export_dv.py --vlora's).  repo: the Hub repository the model
    is ("org/name"), the file's source.repo; None: not said."""
    hf_cfg = _read_config(os.path.join(model_dir, "config.json"))
    tc = _text_config(hf_cfg)
    arch = arch_of(hf_cfg)
    cfg = canonical_target_config(tc) if arch == "qwen35" else canonical_gemma4_config(tc) if arch == "gemma4" \
        else canonical_dense_config(tc)
    if arch == "gemma4" and (draft_dir or dv_dir or lora_dir):
        raise ValueError("a drafter or an adapter beside a Gemma 4 model is not supported yet")
    if arch == "llama" and draft_dir:
        raise ValueError("a block drafter goes in a Qwen3.5 or Qwen3 model's file (beside a Llama: --draft alone)")
    if arch != "llama" and dv_dir:
        raise ValueError("a DV drafter is only supported for Llama (MiniCPM5)")
    if lora_dir and has_lora(dv_dir):
        raise ValueError(f"two V-LoRAs: {dv_dir}'s and {lora_dir}'s")
    lora_dir = lora_dir or (dv_dir if has_lora(dv_dir) else None)
    if lora_dir and arch not in ("llama", "qwen3"):
        raise ValueError("a V-LoRA is only supported for Llama and Qwen3")
    w = Writer()
    quant = _quant_config(model_dir, hf_cfg)
    if activations is not None:
        if activations not in ("nvfp4", "bf16"):
            raise ValueError(f"activations {activations!r}: nvfp4 or bf16")
        if quant is None:
            raise ValueError("activations: the model has no NVFP4 weights")
        quant["activations"] = activations
    seen = set()
    mtp_seen = set()
    with SafeTensors(model_dir) as st:
        for name in st.names():
            if _T_SKIP.match(name):
                continue
            qm = _QUANT_SUFFIX.match(name)
            if qm:  # scales are written with their weight below
                if not quant:
                    raise ValueError(f"{name}: quantization scales in a model without a quantization config")
                continue
            # compressed-tensors' NVFP4 codes: named as the weight they are
            mapped = name[: -len(_PACKED)] + ".weight" if name.endswith(_PACKED) else name
            m = _MTP_PREFIX.match(mapped) if arch == "qwen35" else None
            if m:
                canon = map_name(mapped[m.end():], _MTP_RULES)
                if canon is None:
                    raise ValueError(f"unknown MTP tensor {name!r}")
                _add_linear(w, st, name, "mtp." + canon, cfg)
                mtp_seen.add(canon)
                continue
            if name == "lm_head.weight":
                if cfg["tie_embed"]:
                    continue
                canon = "lm_head"
            else:
                inner = _strip_target_prefix(mapped)
                canon = map_name(inner, _G4_RULES if arch == "gemma4" else _T_RULES) if inner is not None else None
            if canon is None:
                raise ValueError(f"unknown target tensor {name!r}")
            _add_linear(w, st, name, "target." + canon, cfg, reorder_q=arch == "qwen35")
            seen.add(canon)
    if arch == "gemma4":
        _check_gemma4(cfg, seen)
    else:
        _check_target(cfg, seen, qk_norm=arch != "llama")
    mtp = None
    if mtp_seen:
        n_mtp = int(tc.get("mtp_num_hidden_layers", 1))
        _check_mtp(n_mtp, mtp_seen)
        mtp = {"layers": n_mtp}

    draft = _add_draft(w, draft_dir, cfg, log) if draft_dir else None

    tok_path = os.path.join(model_dir, "tokenizer.json")
    tok_info, blobs = export_tokenizer(tok_path)
    for k, v in blobs.items():
        w.add_blob(k, v)
    tok_cfg = _read_optional_config(os.path.join(model_dir, "tokenizer_config.json"))
    template = _chat_template(model_dir, tok_cfg)
    if template:
        w.add_blob("chat_template.jinja", template.encode())
    added = {a["content"]: a["id"] for a in json.loads(blobs["tokenizer.added"])}
    eos = _eos_ids(model_dir, tc, added)
    dv = add_dv(w, dv_dir, cfg, added, log) if dv_dir else None
    lora = add_lora(w, lora_dir, cfg, log) if lora_dir else None
    w.meta = {
        "arch": arch, "config": cfg, "draft": draft, "mtp": mtp, "dv": dv, "lora": lora, "quant": quant,
        "tokenizer": tok_info,
        "generation": {"eos_ids": eos},
        # The engine reads the template (blob above) and recognises its output
        # format itself (models/formats); these are the variables it may print.
        "chat": _template_tokens(tok_cfg),
        "source": {"model": os.path.basename(os.path.normpath(model_dir)), "repo": repo,
                   "base_models": _base_models(model_dir),
                   "draft": os.path.basename(os.path.normpath(draft_dir)) if draft_dir else None,
                   "dv": os.path.basename(os.path.normpath(dv_dir)) if dv_dir else None,
                   "lora": os.path.basename(os.path.normpath(lora_dir)) if lora_dir else None},
    }
    if lora:  # a reader that dropped the adapters would decode the plain model: format 2, which earlier builds refuse
        w.meta["requires"] = ["lora"]
    w.write(out)
    log(f"wrote {out}: {len(w.entries)} entries, draft={draft['kind'] if draft else 'dv' if dv else 'none'}, "
        f"mtp={mtp['layers'] if mtp else 0} layer(s), lora={lora['rank'] if lora else 'none'}, "
        f"weights={quant['format'] if quant else 'as stored'}, eos={eos}")


def _add_draft(w: Writer, draft_dir: str, target: dict | None, log) -> dict:
    """A block drafter's tensors as ``draft.`` names; its canonical config (canonical_draft_config)."""
    draft = canonical_draft_config(_read_config(os.path.join(draft_dir, "config.json")), target)
    dseen = set()
    if draft["kind"] == "dspine":
        with SafeTensors(draft_dir) as dst:
            for canon, dt, arr in dspine_tensors(dst.names(), dst.read, draft, log):
                w.add_tensor("draft." + canon, dt, arr.shape, np.ascontiguousarray(arr))
                dseen.add(canon)
        check_dspine(draft, dseen)
        return draft
    if draft["kind"] == "eagle4":
        with SafeTensors(draft_dir) as dst:
            for canon, dt, arr in eagle4_tensors(dst.names(), dst.read, draft, log):
                w.add_tensor("draft." + canon, dt, arr.shape, np.ascontiguousarray(arr))
                dseen.add(canon)
        check_eagle4(draft, dseen)
        return draft
    with SafeTensors(draft_dir) as dst:
        for name in dst.names():
            inner = name[len("model."):] if name.startswith("model.") else name
            if DRAFT_SKIP.match(inner):
                log(f"draft: ignoring {name} (the engine uses the target's embedding and head)")
                continue
            # DFlash: its layers, fc and norms alone (no tensor of DFlash2's convolutions or selector is read)
            canon = map_name(inner, DRAFT_LAYER_RULES if draft["kind"] == "dflash" else _D_RULES)
            if canon is None:
                raise ValueError(f"unknown draft tensor {name!r}")
            arr, dt = dst.read(name)
            w.add_tensor("draft." + canon, dt, arr.shape, np.ascontiguousarray(arr))
            dseen.add(canon)
    _check_draft(draft, dseen)
    return draft


def convert_draft(draft_dir: str, out: str, log=print) -> None:
    """A drafter (DFlash2, DSpine, EAGLE-4) on its own -> a drafter file: the engine takes it beside a model of any
    format (--drafter-file), and checks it against that model (the "target" its config names)."""
    w = Writer()
    draft = _add_draft(w, draft_dir, None, log)
    w.meta = {"arch": "drafter", "kind": "drafter", "draft": draft,
              "source": {"draft": os.path.basename(os.path.normpath(draft_dir))}}
    w.write(out)
    block = f", block {draft['block']}" if "block" in draft else ""
    log(f"wrote {out}: {len(w.entries)} entries, a {draft['kind']} drafter of {draft['layers']} layer(s){block}, "
        f"taps {draft['taps']}, window {draft['window'] or 'none'}")


def _add_linear(w: Writer, st: SafeTensors, name: str, canon: str, cfg: dict, reorder_q: bool = True) -> None:
    """One tensor; an NVFP4 weight with its two scales.  Qwen3.5 query rows are reordered (weights and scales alike)."""
    arr, dt = st.read(name)
    packed = name.endswith(_PACKED)  # compressed-tensors' NVFP4
    base = name[: -len(_PACKED)] if packed else name[: -len(".weight")] if name.endswith(".weight") else None
    quantized = base is not None and dt == "u8" and (base + ".weight_scale") in st.index
    if canon.endswith("gdn.conv"):
        arr = arr.reshape(arr.shape[0], arr.shape[-1])
    reorder = reorder_q and canon.endswith("attn.q")
    if not quantized:
        if reorder:
            arr = _reorder_q(arr, cfg["n_heads"], cfg["head_dim"])
        w.add_tensor(canon, dt, arr.shape, np.ascontiguousarray(arr))
        return
    sc, sdt = st.read(base + ".weight_scale")
    if packed:  # its global scale is the reciprocal of ModelOpt's: 1 / it, rounded once to float32
        gs, _ = st.read(base + ".weight_global_scale")
        s2 = np.float32(1.0 / float(np.asarray(gs, dtype=np.float64).reshape(-1)[0]))
    else:
        s2, _ = st.read(base + ".weight_scale_2")
    if sdt != "f8e4m3" or sc.shape[1] * 8 != arr.shape[1]:
        raise ValueError(f"{name}: unexpected NVFP4 scale layout {sdt}{list(sc.shape)}")
    if reorder:
        arr = _reorder_q(arr, cfg["n_heads"], cfg["head_dim"])
        sc = _reorder_q(sc, cfg["n_heads"], cfg["head_dim"])
    N, K = arr.shape[0], arr.shape[1] * 2
    w.add_tensor(canon, "nvfp4", [N, K], np.ascontiguousarray(arr))
    w.add_tensor(canon + ".scale", "f8e4m3", sc.shape, np.ascontiguousarray(sc))
    w.add_tensor(canon + ".scale2", "f32", [1], np.ascontiguousarray(np.asarray(s2, dtype=np.float32).reshape(1)))


def _chat_template(model_dir: str, tok_cfg: dict) -> str | None:
    """chat_template.jinja, else tokenizer_config.json's chat_template (the "default" one of a list)."""
    p = os.path.join(model_dir, "chat_template.jinja")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read()
    t = tok_cfg.get("chat_template")
    if isinstance(t, list):
        t = next((x.get("template") for x in t if isinstance(x, dict) and x.get("name") == "default"), None)
    return t if isinstance(t, str) else None


def _template_tokens(tok_cfg: dict) -> dict:
    """bos_token / eos_token as tokenizer_config.json names them (a template may print them)."""
    out = {}
    for k in ("bos_token", "eos_token"):
        v = tok_cfg.get(k)
        v = v.get("content") if isinstance(v, dict) else v
        if isinstance(v, str):
            out[k] = v
    return out


def _eos_ids(model_dir: str, tc: dict, added: dict) -> list[int]:
    ids = []
    for src in (_read_optional_config(os.path.join(model_dir, "generation_config.json")), tc):
        e = src.get("eos_token_id")
        ids += e if isinstance(e, list) else ([e] if e is not None else [])
    for s in ("<|im_end|>", "<|endoftext|>"):
        if s in added:
            ids.append(added[s])
    return sorted(set(int(i) for i in ids))


_GDN_LAYER_KEYS = ("in_norm", "post_norm", "gdn.qkv", "gdn.z", "gdn.b", "gdn.a", "gdn.conv", "gdn.A_log",
                   "gdn.dt_bias", "gdn.norm", "gdn.out", "mlp.gate", "mlp.up", "mlp.down")
_DRAFT_CONV_KEYS = ("attn_conv.base", "attn_conv.proj", "mlp_conv.base", "mlp_conv.proj")



def _check_target(cfg, seen, qk_norm=True):
    """Every tensor the engine reads is there; without q/k norms (Llama), none is either."""
    need = {"embed", "norm"} | ({"lm_head"} if not cfg["tie_embed"] else set())
    for i, t in enumerate(cfg["layer_types"]):
        keys = ATTN_LAYER_KEYS if t == "attn" else _GDN_LAYER_KEYS
        need |= {f"L{i}.{n}" for n in keys if qk_norm or n not in ("attn.q_norm", "attn.k_norm")}
    check_tensors("target", need, seen)
    unread = sorted(n for n in seen - need if n.endswith(("q_norm", "k_norm")))
    if unread:
        raise ValueError(f"q/k norms in a model without them: {unread[:4]}")


def _check_gemma4(cfg, seen):
    """Every tensor Gemma 4's layers read; a global layer's v projection there unless v is k's (k_eq_v)."""
    need = {"embed", "norm"}
    for i, t in enumerate(cfg["layer_types"]):
        need |= {f"L{i}.{n}" for n in _G4_LAYER_KEYS}
        if cfg.get("experts"):
            need |= {f"L{i}.{n}" for n in _G4_MOE_KEYS}
        if t == "sliding" or not cfg["k_eq_v"]:
            need.add(f"L{i}.attn.v")
    check_tensors("target", need, seen)
    extra = sorted(seen - need)
    if extra:
        raise ValueError(f"Gemma 4 tensors the engine would not read: {extra[:4]}")


def _check_mtp(layers, seen):
    need = {"fc", "norm_emb", "norm_hidden", "norm"}
    for j in range(layers):
        need |= {f"L{j}.{n}" for n in ATTN_LAYER_KEYS}
    check_tensors("MTP", need, seen)


def _check_draft(d, seen):
    # The tensors its config says it has (drafter_dir.cpp's check): convolutions where it has taps of them, the
    # selector where it has a rank, its hidden gate where it has one (sel_gate).
    conv, gate = d["conv_taps"] > 0, d["sel_gate"]
    need = {"fc", "hidden_norm", "norm"} | ({"sel.pred", "sel.succ"} if d["sel_rank"] > 0 else set())
    need |= {"sel.hid"} if gate else set()
    for j in range(d["layers"]):
        need |= {f"L{j}.{n}" for n in ((*ATTN_LAYER_KEYS, *_DRAFT_CONV_KEYS) if conv else ATTN_LAYER_KEYS)}
    check_tensors("draft", need, seen)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="flyx_convert", description=__doc__.splitlines()[0])
    ap.add_argument("--model", help="HuggingFace Qwen3.5, Qwen3 or Llama (MiniCPM5) directory (without it: --draft "
                                    "alone, a drafter file for --drafter-file)")
    ap.add_argument("--draft", help="drafter directory: DFlash2 (speculators' or z-lab's), DFlash, DSpark, DSpine or "
                                    "EAGLE-4 (optional)")
    ap.add_argument("--dv", help="DV drafter directory, as python/convert/export_dv.py writes it (optional; Llama), "
                                 "with the V-LoRA it holds")
    ap.add_argument("--lora", help="a V-LoRA alone: the directory export_dv.py --vlora writes (optional; Llama, Qwen3)")
    ap.add_argument("--repo", help="the Hub repository --model is (org/name), kept as the file's source.repo: the "
                                   "drafters made for it are offered as such (optional)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--activations", choices=("nvfp4", "bf16"),
                    help="NVFP4 weights' inputs in prefill passes (default: what the checkpoint is calibrated for)")
    a = ap.parse_args(argv)
    if a.repo and not re.fullmatch(r"[\w.-]+/[\w.-]+", a.repo):
        ap.error("--repo: org/name")

    def log(message):
        # A redirected console may use a code page that cannot print a model's path.
        encoding = sys.stdout.encoding or "utf-8"
        print(message.encode(encoding, errors="backslashreplace").decode(encoding))

    if a.model:
        convert(a.model, a.out, a.draft, log=log, activations=a.activations, dv_dir=a.dv, lora_dir=a.lora,
                repo=a.repo)
    elif a.draft and not a.activations and not a.dv and not a.lora and not a.repo:
        convert_draft(a.draft, a.out, log=log)
    else:
        ap.error("--model, or --draft alone")


if __name__ == "__main__":
    sys.exit(main())
