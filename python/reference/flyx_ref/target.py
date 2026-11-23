# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 text decoder, written from the published equations.

This is the slow, obviously-correct path every engine backend is checked
against.  Tokens are processed one row at a time inside each layer, so the
gated-delta recurrence and the causal conv are the plain textbook form.
Weights use the canonical names produced by ``python/convert``; a linear with
an adapter of its own (``NAME.lora_a`` [rank, in], ``NAME.lora_b`` [out,
rank]: a V-LoRA, python/convert --lora) adds ``lora_scale (x A^T) B^T``.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import torch


@dataclass
class TargetConfig:
    hidden: int
    layers: int
    layer_types: list[str]          # "linear_attention" | "full_attention"
    vocab: int
    inter: int
    n_heads: int
    n_kv: int
    head_dim: int
    rope_theta: float
    rot_dim: int                    # rotated dims at the start of each head
    eps: float
    lin_nk: int
    lin_nv: int
    lin_dk: int
    lin_dv: int
    conv_k: int

    @staticmethod
    def from_canonical(c: dict) -> "TargetConfig":
        lt = ["full_attention" if t == "attn" else "linear_attention" for t in c["layer_types"]]
        return TargetConfig(hidden=c["hidden"], layers=c["layers"], layer_types=lt, vocab=c["vocab"],
                            inter=c["inter"], n_heads=c["n_heads"], n_kv=c["n_kv"], head_dim=c["head_dim"],
                            rope_theta=c["rope_theta"], rot_dim=c["rot_dim"], eps=c["eps"],
                            lin_nk=c["lin_nk"], lin_nv=c["lin_nv"], lin_dk=c["lin_dk"],
                            lin_dv=c["lin_dv"], conv_k=c["conv_k"])

    @property
    def conv_dim(self) -> int:
        return 2 * self.lin_nk * self.lin_dk + self.lin_nv * self.lin_dv


def rms_norm(x, w, eps, offset=0.0):
    """offset=1.0 for Qwen3.5's zero-centred norm, 0.0 for the plain one."""
    y = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)
    return y * (offset + w)


def rope(x, pos, theta, rot_dim):
    """x: [T, heads, hd]; rotate-half RoPE on the first rot_dim dims."""
    half = rot_dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, rot_dim, 2, dtype=torch.float64) / rot_dim))
    ang = pos.to(torch.float64)[:, None] * inv[None, :]          # [T, half]
    cos = torch.cos(ang).to(x.dtype)[:, None, :]
    sin = torch.sin(ang).to(x.dtype)[:, None, :]
    x1, x2, rest = x[..., :half], x[..., half:rot_dim], x[..., rot_dim:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], dim=-1)


def attention(q, k, v, scale, mask=None):
    """q: [Tq, H, d]; k, v: [Tk, Hkv, d]; mask: [Tq, Tk] bool (True = keep)."""
    rep = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    s = torch.einsum("qhd,khd->hqk", q, k) * scale
    if mask is not None:
        s = s.masked_fill(~mask[None], float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("hqk,khd->qhd", p, v)


@dataclass
class TargetState:
    pos: int = 0
    k: dict = field(default_factory=dict)      # layer -> [T, n_kv, hd]
    v: dict = field(default_factory=dict)
    S: dict = field(default_factory=dict)      # layer -> [nv, dk, dv]
    conv: dict = field(default_factory=dict)   # layer -> [conv_k-1, conv_dim] (pre-conv inputs)

    def clone(self) -> "TargetState":
        return copy.deepcopy(self)


def round_bf16(x):
    """Round to bfloat16 (nearest even), as the engine does with the inputs of NVFP4 linears."""
    return x.to(torch.float32).to(torch.bfloat16).to(x.dtype)


class Target:
    def __init__(self, cfg: TargetConfig, W: dict, dtype=torch.float64, a16=(), lora_scale=1.0):
        """a16: canonical names of linears whose inputs the engine rounds to bf16 (NVFP4 weights).  lora_scale: what
        the products of the linears' own adapters are multiplied by (the header's lora section)."""
        self.cfg, self.dtype = cfg, dtype
        self.W = {k: v.to(dtype) for k, v in W.items() if k.startswith("target.")}
        self.a16 = set(a16)
        self.lora_scale = lora_scale

    def w(self, name):
        return self.W["target." + name]

    def _lin(self, x, name):
        a = self.W.get(f"target.{name}.lora_a")
        u = None if a is None else (x @ a.T) * self.lora_scale  # the adapter reads x as it is (bf16 weights)
        if "target." + name in self.a16:
            x = round_bf16(x)
        y = x @ self.w(name).T
        return y if u is None else y + u @ self.w(name + ".lora_b").T

    def new_state(self) -> TargetState:
        c, st = self.cfg, TargetState()
        for i, t in enumerate(c.layer_types):
            if t == "full_attention":
                st.k[i] = torch.zeros(0, c.n_kv, c.head_dim, dtype=self.dtype)
                st.v[i] = torch.zeros(0, c.n_kv, c.head_dim, dtype=self.dtype)
            else:
                st.S[i] = torch.zeros(c.lin_nv, c.lin_dk, c.lin_dv, dtype=self.dtype)
                st.conv[i] = torch.zeros(c.conv_k - 1, c.conv_dim, dtype=self.dtype)
        return st

    # -- mixers ---------------------------------------------------------
    def _attn(self, i, x, st, pos):
        c, p = self.cfg, f"L{i}.attn."
        T = x.shape[0]
        qg = self._lin(x, p + "q")                      # rows: [q heads | gate heads]
        q = qg[:, : c.n_heads * c.head_dim].view(T, c.n_heads, c.head_dim)
        gate = qg[:, c.n_heads * c.head_dim:]
        k = self._lin(x, p + "k").view(T, c.n_kv, c.head_dim)
        v = self._lin(x, p + "v").view(T, c.n_kv, c.head_dim)
        q = rms_norm(q, self.w(p + "q_norm"), c.eps, 1.0)
        k = rms_norm(k, self.w(p + "k_norm"), c.eps, 1.0)
        q = rope(q, pos, c.rope_theta, c.rot_dim)
        k = rope(k, pos, c.rope_theta, c.rot_dim)
        K = torch.cat([st.k[i], k]); V = torch.cat([st.v[i], v])
        st.k[i], st.v[i] = K, V
        P0 = K.shape[0] - T
        mask = torch.arange(K.shape[0])[None, :] <= (P0 + torch.arange(T))[:, None]
        o = attention(q, K, V, 1.0 / math.sqrt(c.head_dim), mask).reshape(T, -1)
        o = o * torch.sigmoid(gate)
        return self._lin(o, p + "o")

    def _gdn(self, i, x, st):
        c, p = self.cfg, f"L{i}.gdn."
        T = x.shape[0]
        mixed = self._lin(x, p + "qkv")                 # [T, conv_dim], pre-conv
        z = self._lin(x, p + "z").view(T, c.lin_nv, c.lin_dv)
        b = self._lin(x, p + "b")
        a = self._lin(x, p + "a")
        win = torch.cat([st.conv[i], mixed])            # [conv_k-1+T, conv_dim]
        cw = self.w(p + "conv")                         # [conv_dim, conv_k]
        conv = torch.stack([(win[t:t + c.conv_k] * cw.T).sum(0) for t in range(T)])
        conv = torch.nn.functional.silu(conv)
        st.conv[i] = win[-(c.conv_k - 1):].clone()
        kd = c.lin_nk * c.lin_dk
        q = conv[:, :kd].view(T, c.lin_nk, c.lin_dk)
        k = conv[:, kd:2 * kd].view(T, c.lin_nk, c.lin_dk)
        v = conv[:, 2 * kd:].view(T, c.lin_nv, c.lin_dv)
        rep = c.lin_nv // c.lin_nk
        q = q.repeat_interleave(rep, dim=1); k = k.repeat_interleave(rep, dim=1)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        q = q / math.sqrt(c.lin_dk)
        beta = torch.sigmoid(b)
        g = -torch.exp(self.w(p + "A_log")) * torch.nn.functional.softplus(a + self.w(p + "dt_bias"))
        S = st.S[i]
        out = torch.empty(T, c.lin_nv, c.lin_dv, dtype=x.dtype)
        for t in range(T):
            S = S * torch.exp(g[t])[:, None, None]
            mem = torch.einsum("hkv,hk->hv", S, k[t])
            delta = (v[t] - mem) * beta[t][:, None]
            S = S + k[t][:, :, None] * delta[:, None, :]
            out[t] = torch.einsum("hkv,hk->hv", S, q[t])
        st.S[i] = S
        o = rms_norm(out, self.w(p + "norm"), c.eps, 0.0) * torch.nn.functional.silu(z)
        return self._lin(o.reshape(T, -1), p + "out")

    # -- full forward over a block of new tokens ---------------------------
    def forward(self, tokens, st: TargetState, taps=()):
        """Advance ``st`` by ``tokens``. Returns (logits [T,V], {n: hidden}, normed hidden)."""
        c = self.cfg
        tokens = torch.as_tensor(tokens, dtype=torch.long)
        T = tokens.shape[0]
        pos = torch.arange(st.pos, st.pos + T)
        x = self.w("embed")[tokens]
        tapped = {}
        for i, t in enumerate(c.layer_types):
            h = rms_norm(x, self.w(f"L{i}.in_norm"), c.eps, 1.0)
            x = x + (self._attn(i, h, st, pos) if t == "full_attention" else self._gdn(i, h, st))
            h = rms_norm(x, self.w(f"L{i}.post_norm"), c.eps, 1.0)
            p = f"L{i}.mlp."
            g = torch.nn.functional.silu(self._lin(h, p + "gate")) * self._lin(h, p + "up")
            x = x + self._lin(g, p + "down")
            if (i + 1) in taps:
                tapped[i + 1] = x.clone()
        st.pos += T
        hn = rms_norm(x, self.w("norm"), c.eps, 1.0)
        head = self.W.get("target.lm_head", self.w("embed"))
        return hn @ head.T, tapped, hn
