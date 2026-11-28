# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""An EAGLE-4 drafter (EAGLE-Corp/Qwen3.8-27B-EAGLE4: ``LlamaForCausalLMEagle4``) -> ``draft.`` tensors.

One Llama decoder layer over two halves side by side: the normalised embedding of a token and the normalised hidden
it continues from.  The hidden is the target's residual stream at a few layers (eagle_aux_hidden_state_layer_ids:
the outputs of those 0-based layers), each tap normalised (fc_norm.<i>) and the taps projected together (fc); a draft
step after the first continues from the previous step's normalised output instead (norm_output).  Its own head
(lm_head) scores drafts, after a rank-r head over the whole vocabulary (efficient_lm_head_down_proj [r, H], then
efficient_lm_head [V, r]) has picked the num_candidates the full head scores (python/reference/flyx_ref/eagle4.py).
Its embed_tokens is a copy of the target's: the engine reads the target's.

The released checkpoint's names map one to one: draft.fc, draft.fc_norm.<i>, draft.L0.{in_norm, hidden_norm,
post_norm, attn.q/k/v/o, mlp.gate/up/down}, draft.norm, draft.head, draft.cand_down, draft.cand_up.  The variant
the released drafter is is the one supported: one layer, per-tap norms, the normalised output carried on, the
low-rank candidate head, the target's vocabulary; anything else is refused, never read as something it is not.
"""
from __future__ import annotations

import re

from .rules import check_tensors

ARCHITECTURE = "LlamaForCausalLMEagle4"

_RULES = [
    (r"fc\.weight", "fc"),
    (r"fc_norm\.(\d+)\.weight", "fc_norm.{0}"),
    (r"layers\.0\.input_layernorm\.weight", "L0.in_norm"),
    (r"layers\.0\.hidden_norm\.weight", "L0.hidden_norm"),
    (r"layers\.0\.post_attention_layernorm\.weight", "L0.post_norm"),
    (r"layers\.0\.self_attn\.(q|k|v|o)_proj\.weight", "L0.attn.{0}"),
    (r"layers\.0\.mlp\.(gate|up|down)_proj\.weight", "L0.mlp.{0}"),
    (r"norm\.weight", "norm"),
    (r"lm_head\.weight", "head"),
    (r"efficient_lm_head_down_proj\.weight", "cand_down"),
    (r"efficient_lm_head\.weight", "cand_up"),
]
_SKIP = re.compile(r"^embed_tokens\.")
LAYER_KEYS = ("in_norm", "hidden_norm", "post_norm", "attn.q", "attn.k", "attn.v", "attn.o", "mlp.gate", "mlp.up",
              "mlp.down")


def _require(ok: bool, what: str) -> None:
    if not ok:
        raise ValueError(f"EAGLE-4 drafter: {what}")


def _int(x) -> bool:
    return type(x) is int  # JSON booleans are a distinct type


def _number(x) -> bool:
    return type(x) in (int, float)


def is_eagle4_config(sc: dict) -> bool:
    return ARCHITECTURE in (sc.get("architectures") or [])


def canonical_eagle4_config(sc: dict, target: dict | None) -> dict:
    """The drafter's config as the engine reads it ("kind": "eagle4"), after checking it is the supported variant.
    taps: the residual stream after that many target layers (eagle_aux_hidden_state_layer_ids + 1).  window: each
    position attends to the positions less than that far before it (0: all).  cand_rank, n_cands: the candidate
    head's rank and the candidates of a row the drafter's head scores exactly."""
    _require(sc.get("architectures") == [ARCHITECTURE], f"architectures {sc.get('architectures')!r}")
    _require(sc.get("model_type") == "llama", f"model_type {sc.get('model_type')!r} (llama)")
    _require(_int(sc.get("num_hidden_layers")) and sc["num_hidden_layers"] == 1,
             f"num_hidden_layers {sc.get('num_hidden_layers')!r} (one layer)")
    _require(sc.get("hidden_act") == "silu", f"hidden_act {sc.get('hidden_act')!r}")
    _require(not sc.get("attention_bias") and not sc.get("mlp_bias"), "attention or MLP biases")
    _require(sc.get("tie_word_embeddings") is False, "tied embeddings (it has a head of its own)")
    _require(type(sc.get("pretraining_tp", 1)) is int and sc.get("pretraining_tp", 1) == 1, "pretraining_tp")
    for key in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "vocab_size"):
        _require(_int(sc.get(key)) and sc[key] > 0, f"{key} {sc.get(key)!r}")
    H, vocab = sc["hidden_size"], sc["vocab_size"]
    heads, kv = sc["num_attention_heads"], sc["num_key_value_heads"]
    hd = sc.get("head_dim", H // heads)
    _require(_int(hd) and hd > 0 and hd % 2 == 0 and heads % kv == 0, f"heads {heads}/{kv}, head_dim {hd!r}")
    _require(_number(sc.get("rms_norm_eps")) and sc["rms_norm_eps"] > 0, "rms_norm_eps")
    _require(_int(sc.get("target_hidden_size", H)) and sc.get("target_hidden_size", H) == H,
             "a target width other than its own (its embedding is the target's)")
    _require(_int(sc.get("draft_vocab_size", vocab)) and sc.get("draft_vocab_size", vocab) == vocab,
             "a draft vocabulary of its own (d2t / t2d)")
    _require(sc.get("fc_norm") is True, "fc_norm must be true (each tap normalised before fc)")
    _require(sc.get("norm_output") is True, "norm_output must be true (a step continues from its normalised output)")
    ec = sc.get("eagle_config")
    _require(isinstance(ec, dict) and ec.get("use_aux_hidden_state") is True, "eagle_config.use_aux_hidden_state")
    ids = ec.get("eagle_aux_hidden_state_layer_ids")
    _require(isinstance(ids, list) and ids and all(_int(i) and i >= 0 for i in ids) and ids == sorted(set(ids)),
             f"eagle_aux_hidden_state_layer_ids {ids!r}")
    n_aux = sc.get("num_aux_hidden_states", len(ids))
    _require(_int(n_aux) and n_aux == len(ids), "num_aux_hidden_states")
    nt = sc.get("num_target_layers")
    _require(nt is None or (_int(nt) and nt > 0), f"num_target_layers {nt!r}")
    if target is not None:
        _require(nt is None or nt == target["layers"], f"made for a model of {nt} layers")
    rp = sc.get("rope_parameters")
    _require(isinstance(rp, dict) and rp.get("rope_type", "default") == "default" and _number(rp.get("rope_theta"))
             and sc.get("rope_scaling") is None and set(rp) <= {"rope_type", "rope_theta"},
             f"rope_parameters {rp!r} (default rotary, its theta)")
    _require(sc.get("rope_is_neox_style", True) is True, "rotary other than NeoX's halves")
    types = sc.get("layer_types")
    _require(isinstance(types, list) and len(types) == 1, f"layer_types {types!r} (one)")
    if types[0] == "sliding_attention":
        _require(sc.get("use_sliding_window") is True and _int(sc.get("sliding_window")) and sc["sliding_window"] > 0,
                 "a sliding layer without its window")
        window = sc["sliding_window"]
    else:
        _require(types[0] == "full_attention" and not sc.get("use_sliding_window"), f"layer type {types[0]!r}")
        window = 0
    head = sc.get("efficient_lm_head_config")
    _require(isinstance(head, dict) and set(head) == {"type", "rank", "num_candidates"} and
             head["type"] == "low_rank" and _int(head["rank"]) and _int(head["num_candidates"]) and
             0 < head["rank"] <= H and 0 < head["num_candidates"] <= vocab,
             f"efficient_lm_head_config {head!r} (low_rank, its rank and candidates)")
    taps = [i + 1 for i in ids]
    if target is not None:
        _require(vocab == target["vocab"], "its vocabulary differs from the target's")
        _require(H == target["hidden"], "its width differs from the target's")
        # the residual after a layer before the last (the engine's taps: models/drafters/eagle4.cpp)
        _require(all(1 <= t < target["layers"] for t in taps), f"taps {taps} outside the target's layers")
    return {
        "kind": "eagle4", "hidden": H, "layers": 1, "n_heads": heads, "n_kv": kv, "head_dim": hd,
        "inter": sc["intermediate_size"], "eps": float(sc["rms_norm_eps"]), "rope_theta": float(rp["rope_theta"]),
        "window": window, "taps": taps, "cand_rank": head["rank"], "n_cands": head["num_candidates"],
        # the layers of the model it was made for, where its config says (the released one does not): the same in a
        # model file and in a drafter file, whose engine checks the taps against the model it is given with
        "target": {"hidden": H, "vocab": vocab, "layers": sc.get("num_target_layers")},
    }


def eagle4_tensors(names, read, d: dict, log):
    """(name, dtype, array) of the engine's tensors, from the checkpoint's `names` read by read(name) -> (array,
    dtype), each checked to have the shape its config gives it."""
    H, n = d["hidden"], len(d["taps"])
    q, kv, V, r = d["n_heads"] * d["head_dim"], d["n_kv"] * d["head_dim"], d["target"]["vocab"], d["cand_rank"]
    shapes = {"fc": [H, n * H], "L0.attn.q": [q, 2 * H], "L0.attn.k": [kv, 2 * H], "L0.attn.v": [kv, 2 * H],
              "L0.attn.o": [H, q], "L0.mlp.gate": [d["inter"], H], "L0.mlp.up": [d["inter"], H],
              "L0.mlp.down": [H, d["inter"]], "head": [V, H], "cand_down": [r, H], "cand_up": [V, r],
              **{k: [H] for k in ("L0.in_norm", "L0.hidden_norm", "L0.post_norm", "norm")},
              **{f"fc_norm.{i}": [H] for i in range(n)}}
    out = []
    for name in names:
        inner = name[len("model."):] if name.startswith("model.") else name
        if _SKIP.match(inner):
            log(f"draft: ignoring {name} (the engine uses the target's embedding)")
            continue
        canon = next((o.format(*m.groups()) for p, o in _RULES if (m := re.fullmatch(p, inner))), None)
        _require(canon is not None and canon in shapes, f"unknown tensor {name!r}")
        arr, dt = read(name)
        _require(list(arr.shape) == shapes[canon], f"{name} {list(arr.shape)} ({shapes[canon]})")
        _require(dt in ("bf16", "f16", "f32"), f"{name}: {dt} (floating point weights)")
        out.append((canon, dt, arr))
    return out


def check_eagle4(d: dict, seen: set[str]) -> None:
    need = {"fc", "norm", "head", "cand_down", "cand_up", *(f"fc_norm.{i}" for i in range(len(d["taps"]))),
            *(f"L0.{k}" for k in LAYER_KEYS)}
    check_tensors("draft", need, seen)
