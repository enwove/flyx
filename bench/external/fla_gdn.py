# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""FLA's gated delta net (flash-linear-attention, Triton) on Qwen3.5-4B's GDN
shapes, timed the way `cuda_bench gdn` times ours: CUDA graphs of 24 layers,
each layer with its own state, so states come from DRAM as in a real pass.

    pip install --target ~/fla-pkg --no-deps fla-core==0.6.0 einops
    PYTHONPATH=~/fla-pkg python bench/external/fla_gdn.py
    ~/flyx-build/bin/cuda_bench gdn        # ours, same shapes

What runs, between the layer's input projections and its output projection:

* plain decoding (1 token): causal_conv1d_update, fused_recurrent_gated_delta_rule
  (gates, beta sigmoid and q/k L2 norm inside the kernel, the new state
  written), rms_norm_gated -- what k_gdn_round does;
* rounds of 2-16 rows (verification): causal_conv1d over the rows,
  fused_recurrent over them without writing a state, rms_norm_gated;
* prefill (1024 tokens): causal_conv1d on q, k and v, chunk_gated_delta_rule
  (gates in the kernel), rms_norm_gated; and the same with FlashQLA's chunked
  kernel (TileLang, github.com/QwenLM/FlashQLA) when it is importable, its
  gates computed before it as it requires:

    pip install --target ~/fqla-pkg --no-deps flash-qla tilelang apache-tvm-ffi \
        torch-c-dlpack-ext cloudpickle ml-dtypes psutil z3-solver
    CUDA_HOME=<toolkit> PYTHONPATH=~/fla-pkg:~/fqla-pkg python bench/external/fla_gdn.py

Inputs are bf16 (FLA's usual path; ours are float32), states float32. Each
path is first checked against a float64 transcription of
python/reference/flyx_ref/target.py's _gdn.
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from fla.modules.causal_conv1d.ops import causal_conv1d, causal_conv1d_update
from fla.modules.norm.fused_norm_gate import rms_norm_gated
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

try:
    import flash_qla
    import flash_qla.ops.gated_delta_rule.chunk as _qla_chunk

    # 0.1.3 on SM120: cp_preprocess_fwd imports aggregate_card_state, which the
    # SM120 branch never defines; its own comment says it is None there.
    if not hasattr(_qla_chunk, "aggregate_card_state"):
        _qla_chunk.aggregate_card_state = None
except ImportError:
    flash_qla = None

NK, NV, DK, DV, CK = 16, 32, 128, 128, 4  # Qwen3.5-4B: key heads, value heads, head dims, conv taps
KD, VD = NK * DK, NV * DV
C = 2 * KD + VD
EPS = 1e-6
dev = "cuda"


class Layer:
    """One GDN layer's parameters and per-sequence state (random, fixed seed)."""

    def __init__(self, gen: torch.Generator):
        r = lambda *s, scale=1.0: (torch.randn(*s, generator=gen, device=dev) * scale)
        self.cw = r(C, CK, scale=0.5).to(torch.bfloat16)
        self.A_log = r(NV, scale=0.5)
        self.dt_bias = r(NV, scale=0.5)
        self.nw = (1.0 + r(DV, scale=0.1)).to(torch.bfloat16)
        self.S = r(1, NV, DK, DV, scale=0.05)
        self.cache = r(1, C, CK, scale=0.5).to(torch.bfloat16)  # the last CK inputs (the conv window)


def inputs(T: int, gen: torch.Generator):
    r = lambda *s: torch.randn(*s, generator=gen, device=dev).to(torch.bfloat16)
    return r(1, T, C), r(1, T, NV, DV), r(1, T, NV), r(1, T, NV)  # mixed (pre-conv), z, b, a


def ref_gdn(L: Layer, mixed, z, b, a, window):
    """target.py _gdn in float64 from the conv input on; window: the CK - 1 inputs before."""
    d = torch.float64
    mixed, z, b, a = mixed[0].to(d), z[0].to(d), b[0].to(d), a[0].to(d)
    T = mixed.shape[0]
    win = torch.cat([window.to(d), mixed])
    cw = L.cw.to(d)
    conv = F.silu(torch.stack([(win[t:t + CK] * cw.T).sum(0) for t in range(T)]))
    q = conv[:, :KD].view(T, NK, DK).repeat_interleave(NV // NK, 1)
    k = conv[:, KD:2 * KD].view(T, NK, DK).repeat_interleave(NV // NK, 1)
    v = conv[:, 2 * KD:].view(T, NV, DV)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / math.sqrt(DK)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    beta = torch.sigmoid(b)
    g = -torch.exp(L.A_log.to(d)) * F.softplus(a + L.dt_bias.to(d))
    S = L.S[0].to(d)
    out = torch.empty(T, NV, DV, dtype=d, device=dev)
    for t in range(T):
        S = S * torch.exp(g[t])[:, None, None]
        delta = (v[t] - torch.einsum("hkv,hk->hv", S, k[t])) * beta[t][:, None]
        S = S + k[t][:, :, None] * delta[:, None, :]
        out[t] = torch.einsum("hkv,hk->hv", S, q[t])
    o = out * torch.rsqrt((out * out).mean(-1, keepdim=True) + EPS) * L.nw.to(d)
    return (o * F.silu(z)).reshape(T, VD), S


def fla_decode(L: Layer, mixed, z, b, a, new_state=True):
    """One token: conv update (moves the window), recurrence, gated norm."""
    x = causal_conv1d_update(mixed, L.cache, weight=L.cw, activation="silu")[0]
    q = x[..., :KD].view(1, 1, NK, DK)
    k = x[..., KD:2 * KD].view(1, 1, NK, DK)
    v = x[..., 2 * KD:].view(1, 1, NV, DV)
    o, S = fused_recurrent_gated_delta_rule(q, k, v, g=a, beta=b, scale=DK ** -0.5, initial_state=L.S,
                                            output_final_state=new_state, use_qk_l2norm_in_kernel=True,
                                            use_gate_in_kernel=True, A_log=L.A_log, dt_bias=L.dt_bias,
                                            use_beta_sigmoid_in_kernel=True)
    return rms_norm_gated(o, z, L.nw, None, activation="swish", eps=EPS).reshape(1, VD), S


def split_conv(L: Layer, mixed, window_state):
    """Three causal convs (q, k, v) as FLA's GatedDeltaNet layer runs them; contiguous outputs.
    window_state: [1, C, CK], the inputs before the rows in its last CK - 1 entries."""
    outs = []
    for lo, hi in ((0, KD), (KD, 2 * KD), (2 * KD, C)):
        xs = mixed[..., lo:hi].contiguous()
        y, _ = causal_conv1d(xs, weight=L.cw[lo:hi], activation="silu",
                             initial_state=window_state[:, lo:hi] if window_state is not None else None)
        outs.append(y)
    return outs


def fla_rows(L: Layer, mixed, z, b, a, init):
    """A round of T rows (no state written) or a prefill chunk (chunked, state written)."""
    T = mixed.shape[1]
    q, k, v = split_conv(L, mixed, init)
    q, k, v = q.view(1, T, NK, DK), k.view(1, T, NK, DK), v.view(1, T, NV, DV)
    kw = dict(g=a, beta=b, scale=DK ** -0.5, initial_state=L.S, use_qk_l2norm_in_kernel=True,
              use_gate_in_kernel=True, A_log=L.A_log, dt_bias=L.dt_bias, use_beta_sigmoid_in_kernel=True)
    if T <= 16:
        o, S = fused_recurrent_gated_delta_rule(q, k, v, output_final_state=False, **kw)
    else:
        o, S = chunk_gated_delta_rule(q, k, v, output_final_state=True, **kw)
    return rms_norm_gated(o, z, L.nw, None, activation="swish", eps=EPS).reshape(T, VD), S


def gates(L: Layer, b, a):
    """beta and the log decay g in float32, for kernels that take them precomputed."""
    return torch.sigmoid(b.float()), -torch.exp(L.A_log) * F.softplus(a.float() + L.dt_bias)


def qla_rows(L: Layer, mixed, z, b, a, init):
    """A prefill chunk with FlashQLA's chunked kernel."""
    T = mixed.shape[1]
    q, k, v = split_conv(L, mixed, init)
    beta, g = gates(L, b, a)
    o, S = flash_qla.chunk_gated_delta_rule(q.view(1, T, NK, DK), k.view(1, T, NK, DK), v.view(1, T, NV, DV), g,
                                            beta, scale=DK ** -0.5, initial_state=L.S, output_final_state=True,
                                            use_qk_l2norm_in_kernel=True)
    return rms_norm_gated(o, z, L.nw, None, activation="swish", eps=EPS).reshape(T, VD), S


def check():
    gen = torch.Generator(device=dev).manual_seed(1)
    L = Layer(gen)
    for T in (1, 8, 64):
        mixed, z, b, a = inputs(T, gen)
        window = L.cache[0, :, 1:].T.float()  # the CK - 1 inputs before this one, oldest first
        y_ref, S_ref = ref_gdn(L, mixed, z, b, a, window)
        if T == 1:
            saved = L.cache.clone()
            y, S = fla_decode(L, mixed, z, b, a)
            L.cache.copy_(saved)
        else:
            y, S = fla_rows(L, mixed, z, b, a, L.cache)
        err = (y.double() - y_ref).abs().max().item() / y_ref.abs().max().item()
        serr = ((S.double() - S_ref).abs().max().item() / S_ref.abs().max().item()) if S is not None else float("nan")
        print(f"check T={T:3d}: output max error {err:.2e} of max |y|, state {serr:.2e}")
        assert err < 3e-2, "FLA call does not compute the reference GDN"
    if flash_qla is not None:
        for T in (64, 1024):
            mixed, z, b, a = inputs(T, gen)
            window = L.cache[0, :, 1:].T.float()
            y_ref, S_ref = ref_gdn(L, mixed, z, b, a, window)
            y, S = qla_rows(L, mixed, z, b, a, L.cache)
            err = (y.double() - y_ref).abs().max().item() / y_ref.abs().max().item()
            serr = (S.double() - S_ref).abs().max().item() / S_ref.abs().max().item()
            print(f"check FlashQLA T={T}: output max error {err:.2e} of max |y|, state {serr:.2e}")
            assert err < 3e-2, "FlashQLA call does not compute the reference GDN"


def graph_us(body, layers: int, reps: int = 50) -> float:
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            for i in range(layers):
                body(i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(layers):
            body(i)
    g.replay()
    torch.cuda.synchronize()
    a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    e.record()
    e.synchronize()
    return a.elapsed_time(e) * 1e3 / reps / layers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--prefill", type=int, default=1024)
    args = ap.parse_args()
    print(torch.cuda.get_device_name(0), "| torch", torch.__version__)
    check()
    gen = torch.Generator(device=dev).manual_seed(2)
    Ls = [Layer(gen) for _ in range(args.layers)]

    X1 = [inputs(1, gen) for _ in Ls]
    t = graph_us(lambda i: fla_decode(Ls[i], *X1[i]), args.layers)
    t_rec = graph_us(lambda i: fused_recurrent_gated_delta_rule(
        X1[i][0][..., :KD].view(1, 1, NK, DK), X1[i][0][..., KD:2 * KD].view(1, 1, NK, DK),
        X1[i][0][..., 2 * KD:].view(1, 1, NV, DV), g=X1[i][3], beta=X1[i][2], scale=DK ** -0.5,
        initial_state=Ls[i].S, output_final_state=True, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        A_log=Ls[i].A_log, dt_bias=Ls[i].dt_bias, use_beta_sigmoid_in_kernel=True), args.layers)
    print(f"plain decode, us per layer: mixer {t:.2f} (recurrence alone {t_rec:.2f})")

    for rows in (2, 4, 8, 16):
        X = [inputs(rows, gen) for _ in Ls]
        t = graph_us(lambda i: fla_rows(Ls[i], *X[i], Ls[i].cache), args.layers)
        qkv = []
        for i, L in enumerate(Ls):
            q, k, v = split_conv(L, X[i][0], L.cache)
            qkv.append((q.view(1, rows, NK, DK), k.view(1, rows, NK, DK), v.view(1, rows, NV, DV)))
        t_rec = graph_us(lambda i: fused_recurrent_gated_delta_rule(
            *qkv[i], g=X[i][3], beta=X[i][2], scale=DK ** -0.5, initial_state=Ls[i].S, output_final_state=False,
            use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, A_log=Ls[i].A_log, dt_bias=Ls[i].dt_bias,
            use_beta_sigmoid_in_kernel=True), args.layers)
        print(f"round of {rows:2d} rows, us per layer: mixer {t:.2f} (recurrence alone {t_rec:.2f})")

    T = args.prefill
    X = [inputs(T, gen) for _ in Ls]
    t = graph_us(lambda i: fla_rows(Ls[i], *X[i], Ls[i].cache), args.layers, reps=10)
    qkv = []
    for i, L in enumerate(Ls):
        q, k, v = split_conv(L, X[i][0], L.cache)
        qkv.append((q.view(1, T, NK, DK), k.view(1, T, NK, DK), v.view(1, T, NV, DV)))
    t_rec = graph_us(lambda i: chunk_gated_delta_rule(
        *qkv[i], g=X[i][3], beta=X[i][2], scale=DK ** -0.5, initial_state=Ls[i].S, output_final_state=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, A_log=Ls[i].A_log, dt_bias=Ls[i].dt_bias,
        use_beta_sigmoid_in_kernel=True), args.layers, reps=10)
    print(f"prefill, {T} tokens, us per layer: mixer {t:.1f} (chunked recurrence alone {t_rec:.1f})")
    if flash_qla is not None:
        t = graph_us(lambda i: qla_rows(Ls[i], *X[i], Ls[i].cache), args.layers, reps=10)
        G = [gates(Ls[i], X[i][2], X[i][3]) for i in range(len(Ls))]
        t_rec = graph_us(lambda i: flash_qla.chunk_gated_delta_rule(
            *qkv[i], G[i][1], G[i][0], scale=DK ** -0.5, initial_state=Ls[i].S, output_final_state=True,
            use_qk_l2norm_in_kernel=True), args.layers, reps=10)
        print(f"prefill, {T} tokens, FlashQLA {flash_qla.__version__}, us per layer: mixer {t:.1f} "
              f"(chunked recurrence alone {t_rec:.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
