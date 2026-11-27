# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""EAGLE-4 drafter (reference).

Behaviour follows the EAGLE-4 paper (*EAGLE-4: One Flexible Drafter for All Batch Sizes and Hardware*, 2026) and
the config of its released drafter (EAGLE-Corp/Qwen3.8-27B-EAGLE4): EAGLE-3's single Llama layer drafting one token
a step, over the target's residual stream at a few layers, with each tap normalised, the layer's output normalised
and carried on, and a candidate head in front of its own full head.  All norms are plain RMSNorms (y = x / rms * w).

The hidden a step continues from: at a position of the sequence, the target's taps there,

    h_j = fc([fc_norm_0(t_1(j)) | ... | fc_norm_{n-1}(t_n(j))])                       (n H -> H)

and after a draft step, that step's normalised output g.  One step at position j, from a hidden h and the token at
j + 1:

    x     = [in_norm(embed(token)) | hidden_norm(h)]                                  (2 H, the embedding the target's)
    q,k,v = x Wq^T, x Wk^T, x Wv^T, q and k rotated at position j (NeoX halves, all of a head's dims)
    r     = h + o(attention(q, the keys at positions j - window < i <= j))            (window 0: all of them)
    r     = r + mlp(post_norm(r))                                                     (SwiGLU)
    g     = norm(r)

Its draft: the candidate head's n_cands best tokens by the rank-r scores (g cand_down^T) cand_up^T (ties toward the
lower id), and of those the one its head scores best, head[c] . g (ties toward the lower id).  The full head's logits
over every token are what a step returns as well, for the tests: the draft is their best only when that token is
among the candidates.

Drafting from anchor a at position p (the hidden at p - 1 known):
    step 1: (h_{p-1}, a)       at position p - 1 -> d_1, g_1
    step i: (g_{i-1}, d_{i-1}) at position p + i - 2 -> d_i
Each step writes its keys and values; after verification the keys and values of the accepted positions are written
again from the target's own taps (the context: (h_j, the token at j + 1) at j), so the layer always attends to the
pairs of the real sequence.  The last position's hidden waits (pending) for the token after it: the next round's
anchor, which its first step pairs it with.  The steps write past the accepted positions too; nothing attends to
those before a later step or context writes them again.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .target import attention, rms_norm, rope


@dataclass
class Eagle4Config:
    hidden: int
    n_heads: int
    n_kv: int
    head_dim: int
    inter: int
    eps: float
    rope_theta: float
    window: int                     # a position attends to the positions less than this far before it (0: all)
    taps: list[int]                 # target residual stream after this many layers
    cand_rank: int
    n_cands: int


def eagle4_config_from_canonical(c: dict) -> Eagle4Config:
    keys = Eagle4Config.__dataclass_fields__.keys()
    return Eagle4Config(**{k: c[k] for k in keys})


@dataclass
class Eagle4State:
    k: dict = field(default_factory=dict)   # position -> [n_kv, hd] (roped)
    v: dict = field(default_factory=dict)
    pos: int = 0                            # positions the context has been given
    pending: torch.Tensor | None = None     # h at pos - 1, waiting for the token at pos


class Eagle4:
    sequential = True                       # drafts one token a step (generate.greedy_spec_eagle4)
    block_drafts = 7                        # the golden rounds' drafts (a sequential drafter's block_drafts: 7)

    def __init__(self, cfg: Eagle4Config, W: dict, dtype=torch.float64):
        self.cfg, self.dtype = cfg, dtype
        self.W = {k[len("draft."):]: v.to(dtype) for k, v in W.items() if k.startswith("draft.")}
        self.embed = W["target.embed"].to(dtype)

    def w(self, name):
        return self.W[name]

    def new_state(self) -> Eagle4State:
        return Eagle4State()

    def hidden(self, feats):
        """h of target rows: feats [R, n H], the taps side by side."""
        c, H = self.cfg, self.cfg.hidden
        parts = [rms_norm(feats[:, i * H:(i + 1) * H], self.w(f"fc_norm.{i}"), c.eps) for i in range(len(c.taps))]
        return torch.cat(parts, dim=-1) @ self.w("fc").T

    def _qkv(self, h, tok: int, pos: int):
        c = self.cfg
        x = torch.cat([rms_norm(self.embed[tok], self.w("L0.in_norm"), c.eps),
                       rms_norm(h, self.w("L0.hidden_norm"), c.eps)])
        q = (x @ self.w("L0.attn.q").T).view(1, c.n_heads, c.head_dim)
        k = (x @ self.w("L0.attn.k").T).view(1, c.n_kv, c.head_dim)
        v = (x @ self.w("L0.attn.v").T).view(1, c.n_kv, c.head_dim)
        pt = torch.tensor([pos])
        return rope(q, pt, c.rope_theta, c.head_dim), rope(k, pt, c.rope_theta, c.head_dim)[0], v[0]

    def context(self, st: Eagle4State, feats, tokens):
        """The target rows at positions st.pos.. (feats [R, n H], their taps; tokens: the token at each): each row's
        token completes the pair of the hidden before it, written at that position; the last row's hidden waits."""
        hs = self.hidden(feats)
        for r in range(len(tokens)):
            if st.pending is not None:
                _, st.k[st.pos - 1], st.v[st.pos - 1] = self._qkv(st.pending, int(tokens[r]), st.pos - 1)
            st.pending = hs[r]
            st.pos += 1

    def step(self, st: Eagle4State, h, tok: int, pos: int):
        """One draft step at `pos` from (h, tok); writes that position's keys and values.  Returns (draft, g, logits
        [vocab] of the full head, candidates [n_cands])."""
        c = self.cfg
        q, st.k[pos], st.v[pos] = self._qkv(h, tok, pos)
        first = max(0, pos - c.window + 1) if c.window else 0
        K = torch.stack([st.k[j] for j in range(first, pos + 1)])
        V = torch.stack([st.v[j] for j in range(first, pos + 1)])
        r = h + attention(q, K, V, 1.0 / math.sqrt(c.head_dim)).reshape(-1) @ self.w("L0.attn.o").T
        t = rms_norm(r, self.w("L0.post_norm"), c.eps)
        r = r + (torch.nn.functional.silu(t @ self.w("L0.mlp.gate").T) * (t @ self.w("L0.mlp.up").T)) @ \
            self.w("L0.mlp.down").T
        g = rms_norm(r, self.w("norm"), c.eps)
        rough = (g @ self.w("cand_down").T) @ self.w("cand_up").T
        cand = rough.argsort(descending=True, stable=True)[:c.n_cands]      # ties toward the lower id
        cand = cand.sort().values                                           # ... and the exact best's ties too
        logits = g @ self.w("head").T
        draft = int(cand[int(torch.argmax(logits[cand]))])
        return draft, g, logits, cand

    def draft(self, st: Eagle4State, anchor: int, width: int, oracle=None):
        """width drafts from the anchor at st.pos (the pending hidden's position + 1).  oracle(pos, token) -> token
        replaces each draft as it is made, and the next step continues from the replacement."""
        p, h, tok, drafts = st.pos, st.pending, anchor, []
        for i in range(width):
            d, h, _, _ = self.step(st, h, tok, p - 1 + i)
            if oracle is not None:
                d = oracle(p + 1 + i, d)
            drafts.append(d)
            tok = d
        return drafts
