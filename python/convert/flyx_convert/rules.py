# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""What the converter's parts share about a checkpoint: a block drafter's Qwen3 layers and its fc and norms as the
engine names them, the tensors a drafter carries for the target's sake (left out), a name mapped by rules, the check
that every tensor the engine reads is there, and the layer fields every block drafter's canonical config has."""
from __future__ import annotations

import re

# A block drafter's (DFlash2's, DSpark's, DSpine's): the context's fc and norms, and each layer's Qwen3 layer.
DRAFT_LAYER_RULES = [
    (r"fc\.weight", "fc"),
    (r"hidden_norm\.weight", "hidden_norm"),
    (r"norm\.weight", "norm"),
    (r"layers\.(\d+)\.input_layernorm\.weight", "L{0}.in_norm"),
    (r"layers\.(\d+)\.post_attention_layernorm\.weight", "L{0}.post_norm"),
    (r"layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight", "L{0}.attn.{1}"),
    (r"layers\.(\d+)\.self_attn\.(q|k)_norm\.weight", "L{0}.attn.{1}_norm"),
    (r"layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight", "L{0}.mlp.{1}"),
]
# A block drafter's copies of what the engine takes from the target (its embedding and head) and training leftovers.
DRAFT_SKIP = re.compile(r"^(embed_tokens\.|lm_head\.|verifier_norm\.|verifier_lm_head\.|t2d|d2t|mask_embedding"
                        r"|confidence_head\.)")
# An attention layer's tensors as the engine names them.
ATTN_LAYER_KEYS = ("in_norm", "post_norm", "attn.q", "attn.k", "attn.v", "attn.o", "attn.q_norm", "attn.k_norm",
                   "mlp.gate", "mlp.up", "mlp.down")


def map_name(name: str, rules) -> str | None:
    """The engine's name for a checkpoint's `name` by the first rule whose pattern it matches whole, else None."""
    for pat, out in rules:
        m = re.fullmatch(pat, name)
        if m:
            return out.format(*m.groups())
    return None


def check_tensors(kind: str, need: set[str], seen: set[str]) -> None:
    """Every tensor the engine reads of `kind` was written."""
    missing = need - seen
    if missing:
        raise ValueError(f"missing {kind} tensors: {sorted(missing)[:10]}")


def layer_fields(tl: dict) -> dict:
    """A transformers layer config's fields every block drafter's canonical config has (drafter_dir.cpp's
    layer_fields): width, layers, heads, head width, MLP width, norm epsilon, rope theta."""
    if not 0 < int(tl["num_hidden_layers"]) <= 1024:
        raise ValueError("draft num_hidden_layers out of range")
    if tl["hidden_size"] <= 0 or tl["num_attention_heads"] <= 0:
        raise ValueError("draft hidden_size and num_attention_heads must be positive")
    hd = tl.get("head_dim") or tl["hidden_size"] // tl["num_attention_heads"]
    if hd <= 0:
        raise ValueError("draft head_dim must be positive")
    rp = tl.get("rope_parameters") or tl.get("rope_scaling") or {}
    return {"hidden": tl["hidden_size"], "layers": tl["num_hidden_layers"], "n_heads": tl["num_attention_heads"],
            "n_kv": tl["num_key_value_heads"], "head_dim": hd, "inter": tl["intermediate_size"],
            "eps": tl["rms_norm_eps"], "rope_theta": float(rp.get("rope_theta", tl.get("rope_theta", 10000.0)))}
