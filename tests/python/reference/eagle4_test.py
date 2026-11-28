# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Exercise the converter and the reference on the tiny EAGLE-4 drafters (tests/e2e/make_tiny.py make_eagle4)."""
import copy
import json
import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(root / "python" / d) for d in ("convert", "reference")]
from flyx_convert.eagle4 import canonical_eagle4_config  # noqa: E402
from flyx_convert.fileformat import Reader  # noqa: E402
from flyx_ref import greedy_plain, greedy_spec  # noqa: E402
from flyx_ref.load import load  # noqa: E402

NAMES = {"fc", "fc_norm.0", "fc_norm.1", "fc_norm.2", "L0.in_norm", "L0.hidden_norm", "L0.post_norm", "L0.attn.q",
         "L0.attn.k", "L0.attn.v", "L0.attn.o", "L0.mlp.gate", "L0.mlp.up", "L0.mlp.down", "norm", "head",
         "cand_down", "cand_up"}


def refused(config, change):
    bad = copy.deepcopy(config)
    change(bad)
    try:
        canonical_eagle4_config(bad, None)
    except (ValueError, KeyError, TypeError):
        return
    raise AssertionError(f"accepted a config it does not read: {bad}")


def check_config(assets):
    config = json.loads((assets / "hf_eagle4" / "config.json").read_text(encoding="utf-8"))
    d = canonical_eagle4_config(config, {"hidden": 64, "vocab": 768, "layers": 8})
    assert d["taps"] == [2, 4, 7] and d["window"] == 0 and d["cand_rank"] == 8 and d["n_cands"] == 16, d
    window = json.loads((assets / "hf_eagle4_window" / "config.json").read_text(encoding="utf-8"))
    assert canonical_eagle4_config(window, None)["window"] == 24
    for change in (
            lambda c: c.update(num_hidden_layers=2),
            lambda c: c.update(num_hidden_layers=True),
            lambda c: c.update(hidden_act="gelu"),
            lambda c: c.update(attention_bias=True),
            lambda c: c.update(mlp_bias=True),
            lambda c: c.update(tie_word_embeddings=True),
            lambda c: c.update(draft_vocab_size=32000),
            lambda c: c.update(target_hidden_size=128),
            lambda c: c.update(fc_norm=False),
            lambda c: c.update(norm_output=False),
            lambda c: c.update(architectures=["LlamaForCausalLMEagle3"]),
            lambda c: c["eagle_config"].update(use_aux_hidden_state=False),
            lambda c: c["eagle_config"].update(eagle_aux_hidden_state_layer_ids=[3, 1]),
            lambda c: c["eagle_config"].update(eagle_aux_hidden_state_layer_ids=[]),
            lambda c: c["efficient_lm_head_config"].update(type="full"),
            lambda c: c["efficient_lm_head_config"].update(rank=True),
            lambda c: c["efficient_lm_head_config"].update(num_candidates=0),
            lambda c: c.pop("efficient_lm_head_config"),
            lambda c: c["rope_parameters"].update(rope_type="yarn"),
            lambda c: c["rope_parameters"].update(factor=4.0),
            lambda c: c.update(rope_is_neox_style=False),
            lambda c: c.update(layer_types=["sliding_attention"]),       # sliding without its window
            lambda c: c.update(layer_types=["full_attention", "full_attention"]),
    ):
        refused(config, change)
    for target in ({"hidden": 32, "vocab": 768, "layers": 8}, {"hidden": 64, "vocab": 700, "layers": 8},
                   {"hidden": 64, "vocab": 768, "layers": 6}):
        try:
            canonical_eagle4_config(config, target)
        except ValueError:
            continue
        raise AssertionError(f"accepted a drafter made for another target than {target}")


def check_tensors(assets):
    for name in ("tiny_eagle4.flyx", "tiny_eagle4_window.flyx", "tiny_draft_eagle4.flyx",
                 "tiny_draft_eagle4_window.flyx"):
        with Reader(str(assets / name)) as r:
            names = {n[len("draft."):] for n in r.header["tensors"] if n.startswith("draft.")}
            assert names == NAMES, (name, sorted(names ^ NAMES))
            assert r.header["draft"]["kind"] == "eagle4"
            shapes = {n: r.header["tensors"]["draft." + n]["shape"] for n in ("fc", "L0.attn.q", "cand_up")}
            assert shapes == {"fc": [64, 192], "L0.attn.q": [128, 128], "cand_up": [768, 8]}, shapes


def check_reference(assets):
    prompts = [[5, 17, 33, 9], [7, 3, 99, 12, 1, 2]]
    for name in ("tiny_eagle4.flyx", "tiny_eagle4_window.flyx"):
        target, draft, _, _ = load(str(assets / name))
        for pr in prompts:
            plain, _ = greedy_plain(target, pr, 40)
            for width in (1, 3, 7, 15):
                spec, rounds = greedy_spec(target, draft, pr, 40, width=width)
                assert spec == plain and all(len(r["drafts"]) == width for r in rounds), (name, width)
            # drafts from the true continuation, wrong at every fifth position: rounds accept 0..width drafts
            arr = list(pr) + plain
            arr = [t if i < len(pr) or i % 5 != 2 else (t + 1) % target.cfg.vocab for i, t in enumerate(arr)]
            oracle = lambda pos, d, arr=arr: arr[pos] if pos < len(arr) else 0  # noqa: E731
            spec, rounds = greedy_spec(target, draft, pr, 40, width=7, oracle=oracle)
            assert spec == plain and max(r["n"] for r in rounds) >= 3, [r["n"] for r in rounds]


def check_candidates(assets):
    """The draft is the head's best of the candidates: the full head's best when it is among them, else not."""
    target, draft, _, _ = load(str(assets / "tiny_eagle4.flyx"))
    st = draft.new_state()
    pr = [5, 17, 33, 9, 300, 301, 302, 303]
    _, tapped, _ = target.forward(pr, target.new_state(), taps=draft.cfg.taps)
    draft.context(st, torch.cat([tapped[t] for t in draft.cfg.taps], dim=-1), pr)
    among = missed = 0
    h, tok = st.pending, 42
    for i in range(32):
        d, h, logits, cand = draft.step(st, h, tok, st.pos - 1 + i)
        best = int(torch.argmax(logits))
        assert d in cand.tolist() and logits[d] == logits[cand].max()
        among += best in cand.tolist()
        missed += best not in cand.tolist()
        assert (best == d) == (best in cand.tolist())
        tok = d
    assert among > 0 and missed > 0, f"the head's best among the candidates at {among} steps, not at {missed}"
    # every rough score tied: the candidates are the lowest ids
    draft.W["cand_up"] = torch.zeros_like(draft.W["cand_up"])
    _, _, _, cand = draft.step(st, h, tok, st.pos - 1 + 32)
    assert cand.tolist() == list(range(draft.cfg.n_cands))
    return among, missed


def check_window(assets):
    """A window wider than the sequence is no window; a narrower one changes the drafts past it."""
    target, draft, _, _ = load(str(assets / "tiny_eagle4_window.flyx"))
    pr = [5, 17, 33, 9]
    plain, _ = greedy_plain(target, pr, 40)

    def drafts(window):
        draft.cfg.window = window
        return [r["drafts"] for r in greedy_spec(target, draft, pr, 40, width=3)[1]]

    full, wide, narrow = drafts(0), drafts(1000), drafts(24)
    assert full == wide
    assert narrow[:5] == full[:5] and narrow != full  # the first rounds' positions sit inside the window


def main(assets):
    check_config(assets)
    check_tensors(assets)
    check_reference(assets)
    among, missed = check_candidates(assets)
    check_window(assets)
    print(f"eagle4 reference: ok (the head's best among the candidates at {among} of {among + missed} steps)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
