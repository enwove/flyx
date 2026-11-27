# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The DSpark reference (python/reference/flyx_ref/draft.py, kind "dspark") against DeepSpec's own model.

    python tests/python/reference/check_dspark_deepspec.py DEEPSPEC ASSETS

Run by hand: DEEPSPEC is a checkout of deepseek-ai/DeepSpec (its deepspec/ package and transformers 5); ASSETS is a
build's tiny assets (make_tiny.py): tiny_dspark.flyx, the tiny Qwen3.5 with a DSpark drafter in its file, and
hf_dspark/, that drafter as DeepSpec saves one.  DeepSpec's Qwen3DSparkModel is given the target's embedding and head
(its frozen copies are the target's) and the reference's context features; for a few prompts, float64:

1. every row's final hidden state (DeepSpec's _forward_backbone over the context and the block [anchor, mask...]);
2. the Markov-corrected logits of the head's candidates against the reference's walk scores;
3. the drafts, the reference's walk over the whole vocabulary (sel_topk = vocab) against DeepSpec's greedy
   sample_draft_tokens.
"""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python", "reference"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python", "convert"))


def main(deepspec: str, assets: str) -> int:
    sys.path.insert(0, deepspec)
    from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel  # noqa: E402
    from flyx_ref.generate import _feats  # noqa: E402
    from flyx_ref.load import load  # noqa: E402

    target, draft, _, _ = load(os.path.join(assets, "tiny_dspark.flyx"))
    assert draft is not None and draft.cfg.draft_from == 0 and not draft.cfg.sel_gate, \
        "tiny_dspark.flyx holds no DSpark drafter (every row a draft, no selector gate)"
    draft.cfg.sel_topk = draft.head.shape[0]  # the walk over the whole vocabulary: DSpark's exact greedy
    m = Qwen3DSparkModel.from_pretrained(os.path.join(assets, "hf_dspark"), dtype=torch.float64,
                                         attn_implementation="eager").eval()
    with torch.no_grad():
        m.embed_tokens.weight.copy_(draft.embed)
        m.lm_head.weight.copy_(draft.head)
    worst_hidden = worst_score = 0.0
    same = total = 0
    for prompt in ([5, 17, 33, 9], [7, 3, 99, 12, 40, 41, 77, 500, 2, 9, 10], [300, 301, 302]):
        st = target.new_state()
        logits, taps, _ = target.forward(prompt, st, taps=draft.cfg.taps)
        feats = _feats(taps, draft.cfg.taps)
        anchor = int(torch.argmax(logits[-1]))
        ds = draft.new_state()
        draft.append_context(ds, feats)
        ref, dbg = draft.block(ds, anchor)
        B, ctx = m.block_size, feats.shape[0]
        ids = torch.tensor([[anchor] + [m.mask_token_id] * (B - 1)])
        with torch.no_grad():
            hid = m._forward_backbone(position_ids=torch.arange(ctx + B)[None], noise_embedding=m.embed_tokens(ids),
                                      target_hidden_states=feats[None].to(torch.float64), attention_mask=None,
                                      is_causal=False)
            toks, _ = m.sample_draft_tokens(m.compute_logits(hid), first_prev_token_ids=ids[:, 0], temperature=0.0,
                                            hidden_states=hid)
            prev = torch.tensor([anchor] + ref[:-1])
            corrected = m.compute_logits(hid[0]) + m.markov_head.compute_step_bias(prev, None)
        scale = float(hid.abs().max())
        worst_hidden = max(worst_hidden, float((hid[0] - dbg["hidden"]).abs().max()) / scale)
        worst_score = max(worst_score, float((corrected.gather(1, dbg["cand"]) - dbg["scores"]).abs().max()))
        same += sum(a == b for a, b in zip(ref, toks[0].tolist()))
        total += len(ref)
    print(f"hidden states: largest difference {worst_hidden:.2e} of their scale; walk scores: {worst_score:.2e}; "
          f"drafts {same} / {total} the same")
    ok = worst_hidden < 1e-5 and worst_score < 1e-3 and same == total
    print("PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
