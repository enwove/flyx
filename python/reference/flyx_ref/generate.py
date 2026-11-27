# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Greedy generation loops: plain, and verified with a DFlash2 block draft.

Both must emit exactly the same tokens; the speculative loop also records
the per-round drafts and acceptance so engine rounds can be compared one by
one.
"""
from __future__ import annotations

import torch

from .draft import Draft
from .target import Target


def _feats(taps: dict, ids: list[int]) -> torch.Tensor:
    return torch.cat([taps[i] for i in ids], dim=-1)


def greedy_plain(target: Target, prompt, max_new, stop_ids=(), presence=0.0, frequency=0.0):
    """presence / frequency: what a token already chosen in this reply loses from its logit, once
    and per time it was chosen (ops/sampling/sampling.h); margins are those of the penalized logits."""
    st = target.new_state()
    counts: dict[int, int] = {}

    def choose(row):
        row = row.clone()
        for t, c in counts.items():
            row[t] -= presence + frequency * c
        tok = int(torch.argmax(row))
        counts[tok] = counts.get(tok, 0) + 1
        return tok, _margin(row)

    logits, _, _ = target.forward(prompt, st)
    tok, margin = choose(logits[-1])
    out, margins = [tok], [margin]
    while len(out) < max_new and out[-1] not in stop_ids:
        logits, _, _ = target.forward([out[-1]], st)
        tok, margin = choose(logits[-1])
        out.append(tok)
        margins.append(margin)
    return out, margins


def _margin(row):
    top2 = torch.topk(row, 2).values
    return float(top2[0] - top2[1])


def greedy_spec(target: Target, draft: Draft, prompt, max_new, stop_ids=(), width=None, oracle=None):
    """width: drafts verified per round (<= block - draft_from). oracle(pos)->token|None overrides drafts (tests)."""
    c = draft.cfg
    K = width or (c.block - c.draft_from)
    st = target.new_state()
    ds = draft.new_state()
    logits, taps, _ = target.forward(prompt, st, taps=c.taps)
    draft.append_context(ds, _feats(taps, c.taps))
    out = [int(torch.argmax(logits[-1]))]
    rounds = []
    while len(out) < max_new and out[-1] not in stop_ids:
        anchor, p = out[-1], st.pos                     # anchor sits at position p
        drafts, _ = draft.block(ds, anchor)
        drafts = drafts[:K]
        if oracle is not None:
            drafts = [oracle(p + 1 + i, d) for i, d in enumerate(drafts)]
        rows = [anchor] + drafts
        trial = st.clone()
        logits, _, _ = target.forward(rows, trial)
        choice = [int(x) for x in torch.argmax(logits, dim=-1)]
        n = 0
        while n < K and drafts[n] == choice[n] and choice[n] not in stop_ids:
            n += 1
        emit = choice[: n + 1]
        # commit: rows 0..n become target state and draft context
        _, taps, _ = target.forward(rows[: n + 1], st, taps=c.taps)
        draft.append_context(ds, _feats(taps, c.taps))
        rounds.append({"pos": p, "anchor": anchor, "drafts": drafts, "choice": choice, "n": n})
        for t in emit:
            out.append(t)
            if t in stop_ids or len(out) >= max_new:
                break
    return out[:max_new], rounds


def greedy_spec_mtp(target: Target, mtp, prompt, max_new, width=3, stop_ids=(), oracle=None):
    """Greedy decoding verified with drafts from the model's own MTP layer.

    oracle(pos, token) -> token replaces each draft as it is made (and feeds
    the replacement to the next draft step), as the engine's test hook does.
    """
    st = target.new_state()
    ms = mtp.new_state()
    logits, _, hn = target.forward(prompt, st)
    mtp.context(ms, hn[:-1], list(prompt[1:]), 0)
    pending = hn[-1]                                    # target hidden at the last processed position
    out = [int(torch.argmax(logits[-1]))]
    rounds = []
    while len(out) < max_new and out[-1] not in stop_ids:
        anchor, p = out[-1], st.pos
        drafts, h, tok = [], pending, anchor
        for i in range(width):
            lg, h = mtp.step(ms, h, tok, p - 1 + i)
            d = int(torch.argmax(lg))
            if oracle is not None:
                d = oracle(p + 1 + i, d)
            drafts.append(d)
            tok = d
        rows = [anchor] + drafts
        trial = st.clone()
        logits, _, _ = target.forward(rows, trial)
        choice = [int(x) for x in torch.argmax(logits, dim=-1)]
        n = 0
        while n < width and drafts[n] == choice[n] and choice[n] not in stop_ids:
            n += 1
        _, _, hn = target.forward(rows[: n + 1], st)
        mtp.context(ms, hn[:n], rows[1: n + 1], p)
        pending = hn[n]
        rounds.append({"pos": p, "anchor": anchor, "drafts": drafts, "choice": choice, "n": n})
        for t in choice[: n + 1]:
            out.append(t)
            if t in stop_ids or len(out) >= max_new:
                break
    return out[:max_new], rounds
