# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Qwen3.5's multi-token-prediction layer as a drafter (reference).

One decoder layer that predicts the token after next.  Its input at position
j combines the target's normalised hidden state at j with the embedding of
the token at j + 1:

    x_j = fc([norm_e(embed(t_{j+1})), norm_h(h_j)])     (2H -> H)
    g_j = norm(layer(x_j))                              (gated attention over its own KV + MLP)
    logits = lm_head(g_j)                               (shared with the target)

Drafting K tokens from anchor a at position p (h_{p-1} known):
    step 1: (h_{p-1}, a)   at position p-1 -> d_1
    step i: (g_{i-1}, d_{i-1}) at position p+i-2 -> d_i
After verification the KV entries of the accepted positions are rewritten
from the target's own hidden states (the context pass), so the layer always
attends to (h_j, t_{j+1}) pairs of the real sequence.  All norms are
Qwen3.5's zero-centred ones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .target import TargetConfig, attention, rms_norm, rope


@dataclass
class MtpState:
    k: dict = field(default_factory=dict)   # position -> [n_kv, hd]
    v: dict = field(default_factory=dict)


class Mtp:
    def __init__(self, cfg: TargetConfig, W: dict, dtype=torch.float64):
        self.cfg, self.dtype = cfg, dtype
        self.W = {k: v.to(dtype) for k, v in W.items() if k.startswith("mtp.")}
        self.embed = W["target.embed"].to(dtype)
        self.head = W.get("target.lm_head", W["target.embed"]).to(dtype)

    def w(self, name):
        return self.W["mtp." + name]

    def new_state(self) -> MtpState:
        return MtpState()

    def _input(self, h, tok):
        c = self.cfg
        e = rms_norm(self.embed[tok], self.w("norm_emb"), c.eps, 1.0)
        hh = rms_norm(h, self.w("norm_hidden"), c.eps, 1.0)
        return torch.cat([e, hh], dim=-1) @ self.w("fc").T

    def _qkv(self, x, pos):
        c, p = self.cfg, "L0.attn."
        t = rms_norm(x, self.w("L0.in_norm"), c.eps, 1.0)
        qg = t @ self.w(p + "q").T
        q = qg[: c.n_heads * c.head_dim].view(1, c.n_heads, c.head_dim)
        gate = qg[c.n_heads * c.head_dim:]
        k = (t @ self.w(p + "k").T).view(1, c.n_kv, c.head_dim)
        v = (t @ self.w(p + "v").T).view(1, c.n_kv, c.head_dim)
        q = rms_norm(q, self.w(p + "q_norm"), c.eps, 1.0)
        k = rms_norm(k, self.w(p + "k_norm"), c.eps, 1.0)
        pt = torch.tensor([pos])
        return rope(q, pt, c.rope_theta, c.rot_dim), rope(k, pt, c.rope_theta, c.rot_dim)[0], v[0], gate

    def context(self, st: MtpState, hs, next_tokens, pos0: int):
        """KV entries at pos0 + r from (hs[r], next_tokens[r])."""
        for r in range(len(next_tokens)):
            _, k, v, _ = self._qkv(self._input(hs[r], int(next_tokens[r])), pos0 + r)
            st.k[pos0 + r], st.v[pos0 + r] = k, v

    def step(self, st: MtpState, h, tok: int, pos: int):
        """One draft step at `pos`; writes that KV entry.  Returns (logits, g)."""
        c, p = self.cfg, "L0."
        x = self._input(h, tok)
        q, k, v, gate = self._qkv(x, pos)
        st.k[pos], st.v[pos] = k, v
        K = torch.stack([st.k[j] for j in range(pos + 1)])
        V = torch.stack([st.v[j] for j in range(pos + 1)])
        o = attention(q, K, V, 1.0 / math.sqrt(c.head_dim)).reshape(-1)
        o = o * torch.sigmoid(gate)
        x = x + o @ self.w(p + "attn.o").T
        t = rms_norm(x, self.w(p + "post_norm"), c.eps, 1.0)
        x = x + (torch.nn.functional.silu(t @ self.w(p + "mlp.gate").T) * (t @ self.w(p + "mlp.up").T)) @ \
            self.w(p + "mlp.down").T
        g = rms_norm(x, self.w("norm"), c.eps, 1.0)
        return g @ self.head.T, g
