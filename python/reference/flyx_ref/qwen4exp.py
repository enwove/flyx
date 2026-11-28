# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Teacher-forced Qwen4Exp text equations on CPU doubles and original checkpoint weights."""
import math

import torch
from torch.nn import functional as F

from .target import attention, rms_norm, rope


def _norm(x, module):
    shape = x.shape[-1:] if module.group_size is None else x.shape[-2:]
    return rms_norm(x, module.weight.double().reshape(shape), module.eps, 1.0)


def _linear(x, module):
    return F.linear(x, module.weight.double())


def _mix(wide, hc):
    normalized = _norm(wide, hc.hc_norm)
    flat = normalized.flatten(1)
    streams = wide.shape[1]
    low = F.silu(_linear(flat, hc.input_mix_weight_down) / streams)
    selector = _linear(low, hc.input_mix_weight_up).sigmoid().reshape_as(wide)
    body = (selector * normalized).mean(1)
    inject = hc.block_inject_weight
    return body, None if inject is None else 2 * (_linear(flat, inject) / streams).sigmoid().unsqueeze(-1)


def _causal(x, module, dilation=1):
    channels = x.shape[-1]
    weight = module.weight.double()
    padded = F.pad(x.T.unsqueeze(0), ((weight.shape[-1] - 1) * dilation, 0))
    return F.conv1d(padded, weight, groups=channels, dilation=dilation)[0].T


def _gdn(x, g):
    qkv = F.silu(_causal(_linear(x, g.in_proj_qkv), g.conv1d))
    q, k, v = qkv.split([g.key_dim, g.key_dim, g.value_dim], -1)
    dim, vdim = g.head_k_dim, g.head_v_dim
    v = v.reshape(len(x), -1, vdim)
    def heads(z):
        z = z.reshape(len(x), -1, dim)
        z = z * torch.rsqrt(z.square().sum(-1, keepdim=True) + 1e-6)
        return z.repeat_interleave(v.shape[1] // z.shape[1], 1)
    q, k = heads(q) / math.sqrt(dim), heads(k)
    decay = (-g.A_log.double().exp() * F.softplus(_linear(x, g.in_proj_a) + g.dt_bias.double())).exp()
    beta = _linear(x, g.in_proj_b).sigmoid()
    state = x.new_zeros(v.shape[1], dim, vdim)
    outputs = []
    for t in range(len(x)):
        state = state * decay[t, :, None, None]
        error = v[t] - torch.einsum("hd,hdv->hv", k[t], state)
        state += k[t, :, :, None] * (beta[t, :, None] * error)[:, None, :]
        outputs.append(torch.einsum("hd,hdv->hv", q[t], state))
    z = _linear(x, g.in_proj_z).reshape_as(v)
    y = rms_norm(torch.stack(outputs), g.norm.weight.double(), g.norm.variance_epsilon)
    return _linear((y * z.sigmoid()).flatten(1), g.out_proj)


def _attention(x, a, cfg):
    t, dim = len(x), a.head_dim
    pos = torch.arange(t)
    rp = cfg.rope_parameters
    rot = int(dim * rp["partial_rotary_factor"])
    def rotate(z, p):
        return rope(z, p, rp["rope_theta"], rot)
    q, gate = _linear(x, a.q_proj).reshape(t, -1, 2 * dim).chunk(2, -1)
    q = rotate(_norm(q, a.q_norm), pos)
    k = rotate(_norm(_linear(x, a.k_proj).reshape(t, -1, dim), a.k_norm), pos)
    v = _linear(x, a.v_proj).reshape_as(k)
    mask = pos[:, None] >= pos
    ix = a.indexer
    group = ix.compress_ratio
    if t >= (ix.block_topk + 1) * group:
        d = ix.index_head_dim
        iq, raw = _linear(x, ix.index_qk_proj).reshape(t, -1, d).split([ix.index_n_heads, ix.index_kv_heads], 1)
        iq = rotate(_norm(iq, ix.q_layernorm), pos)
        starts = torch.arange(0, t - group + 1, group)
        pooled = raw[:len(starts) * group].reshape(len(starts), group, ix.index_kv_heads, d).mean(1)
        keys = rotate(_norm(pooled, ix.k_layernorm), starts)
        for i in range(t):
            count = (i + 1) // group
            if count <= ix.block_topk:
                continue
            scores = (iq[i] @ keys[:count, 0].T).relu().sum(0)
            keep = scores.argsort(descending=True, stable=True)[:ix.block_topk]
            mask[i, :count * group] = False
            mask[i, (group * keep[:, None] + torch.arange(group)).flatten()] = True
    y = attention(q, k, v, dim ** -0.5, mask) * gate.sigmoid()
    return _linear(y.flatten(1), a.o_proj)


def q8_inputs(x):
    """GGML's per-block activation rounding before a packed weight projection."""
    blocks = x.float().reshape(*x.shape[:-1], -1, 32)
    scale = blocks.abs().amax(-1, keepdim=True) / 127
    codes = (blocks / torch.where(scale == 0, 1, scale)).round()
    return (codes.to(x.dtype) * scale.to(x.dtype)).reshape_as(x)


def _experts(x, m, q8):
    vals, ids = _linear(x, m.gate).topk(m.gate.top_k)
    weights = vals.softmax(-1)
    routed = q8_inputs(x) if q8 else x
    out = x.new_zeros(x.shape)
    for r in range(len(x)):
        for expert, weight in zip(ids[r], weights[r]):
            g, u = (m.experts.gate_up_proj[expert].double() @ routed[r]).chunk(2)
            out[r] += weight * (m.experts.down_proj[expert].double() @ (F.silu(g) * u))
    s = m.shared_expert
    shared = _linear(F.silu(_linear(x, s.gate_proj)) * _linear(x, s.up_proj), s.down_proj)
    return out + _linear(x, m.shared_expert_gate).sigmoid() * shared


def _ple(wide, tokens, b):
    e = b.ple_embedding
    mul = e.layer_multipliers.tolist()
    offsets, sizes = e.ngram_heads_offsets.tolist(), e.ngram_heads_vocab_sizes.tolist()
    history, ids = [], []
    for token in tokens:
        neighbors = [token] + (history[::-1] + [e.eos_token_id] * len(mul))[:len(mul) - 1]
        mixed, row = neighbors[0] * mul[0], []
        for order in range(1, len(mul)):
            mixed ^= neighbors[order] * mul[order]
            for sub in range(e.heads_per_ngram):
                h = (order - 1) * e.heads_per_ngram + sub
                row.append(offsets[h] + mixed % sizes[h])
        ids.append(row)
        history = [] if token == e.eos_token_id else (history + [token])[-len(mul):]
    embed = e.ngram_embedding.weight[torch.tensor(ids)].double().flatten(1)
    k = _norm(_linear(embed, b.key_proj).reshape_as(wide), b.norm_key)
    alignment = (_norm(wide, b.norm_query) * k).sum(-1, keepdim=True) / math.sqrt(wide.shape[-1])
    gate = (alignment.sign() * alignment.abs().clamp_min(1e-6).sqrt()).sigmoid()
    y = gate * _linear(embed, b.value_proj)[:, None, :]
    conv = _causal(_norm(y, b.norm_conv).flatten(1), b.conv1d, b.conv1d.dilation[0])
    return wide + (y + F.silu(conv).reshape_as(wide))


@torch.inference_mode()
def logits(model, tokens, q8_expert_layers=()):
    cfg, trunk = model.config, model.model
    x = trunk.embed_tokens.weight[tokens].double()[:, None, :].expand(-1, cfg.hc_count, -1)
    for index, layer in enumerate(trunk.layers):
        if layer.ple is not None:
            x = _ple(x, tokens, layer.ple)
        body, inject = _mix(x, layer.attn_hyper_connection)
        y = (_gdn(body, layer.linear_attn) if layer.layer_type == "linear_attention"
             else _attention(body, layer.self_attn, cfg))
        x = x + inject * y[:, None, :]
        body, inject = _mix(x, layer.mlp_hyper_connection)
        x = x + inject * _experts(body, layer.mlp, index in q8_expert_layers)[:, None, :]
    out, _ = _mix(x, trunk.hyper_connection_mixer)
    return _linear(out, model.lm_head)

