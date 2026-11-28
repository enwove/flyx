# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""A tiny gpt-oss as llama.cpp's converter lays it out (arch "gpt-oss"), and Hugging Face's own outputs for it.

    tiny_gpt_oss.gguf     4 layers (sliding, full, sliding, full; a window of 8), 4 query heads of 32 over 2 KV heads
                          on a hidden size of 64, 4 experts of 96 (2 a token) in MXFP4, attention and the embedding
                          and head in Q8_0, the router, biases, sinks and norms in float32; YaRN rotary
    golden_gpt_oss.json   transformers' GptOssForCausalLM in float64 on the weights the file holds (every block
                          decoded back into it), greedy: each case's tokens and the smallest margin between the best
                          two logits along them; the cases kept have one of at least MIN_MARGIN, and every router of
                          every position a gap of at least MIN_ROUTER_GAP between the logits of its k-th and (k+1)-th
                          experts (an engine's rounding then moves no token and no expert)

Its vocabulary: the 256 bytes as GPT-2 writes them, harmony's special tokens, reserved ones up to 512.
"""
from __future__ import annotations

import hashlib
import json
import os
from importlib import metadata
from pathlib import Path

import gguf
import numpy as np

H, LAYERS, NH, NKV, HD, E, TOP_K, INTER, WINDOW, VOCAB = 64, 4, 4, 2, 32, 4, 2, 96, 8, 512
THETA, EPS, FACTOR, ORIG_CTX, BETA_FAST, BETA_SLOW = 150000.0, 1e-5, 32.0, 4096, 32.0, 1.0
SPECIALS = ["<|startoftext|>", "<|endoftext|>", "<|return|>", "<|constrain|>", "<|channel|>", "<|start|>", "<|end|>",
            "<|message|>", "<|call|>"]
MIN_MARGIN, MIN_ROUTER_GAP, NEW_TOKENS, CASES = 0.15, 0.05, 12, 8


def byte_tokens():
    """GPT-2's printable stand-ins for the 256 bytes (its bytes_to_unicode)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    table = dict(zip(bs, cs))
    return [chr(table[b]) for b in range(256)]


def quantized(a, typ):
    """a's rows in GGML blocks of type typ, and the values they decode to (typ None: float32 as it is)."""
    if typ is None:
        return a.astype(np.float32), a.astype(np.float32)
    q = gguf.quants.quantize(np.ascontiguousarray(a, dtype=np.float32), typ)
    return q, gguf.quants.dequantize(q, typ).reshape(a.shape).astype(np.float32)


def inputs_digest() -> str:
    """What the files are made from: this script and the packages that make them."""
    h = hashlib.sha256(Path(__file__).read_bytes())
    for pkg in ("torch", "transformers", "gguf", "numpy"):
        h.update(f"{pkg} {metadata.version(pkg)}\0".encode())
    return h.hexdigest()


def write(out: Path, quant: bool = True, stem: str = "gpt_oss"):
    """quant False: every tensor float32 (no input rounding anywhere).  Skipped where the files are there, made from
    the same inputs (their stamp)."""
    stamp = out / f"tiny_{stem}.stamp"
    if stamp.exists() and stamp.read_text() == inputs_digest() and (out / f"tiny_{stem}.gguf").exists() and \
            (out / f"golden_{stem}.json").exists():
        return
    import torch
    from transformers import GptOssConfig, GptOssForCausalLM

    torch.set_num_threads(min(4, os.cpu_count() or 1))  # a tiny model's products: more threads only spin
    torch.manual_seed(20261127)
    rng = np.random.default_rng(20261127)
    cfg = GptOssConfig(
        num_hidden_layers=LAYERS, num_local_experts=E, vocab_size=VOCAB, hidden_size=H, intermediate_size=INTER,
        head_dim=HD, num_attention_heads=NH, num_key_value_heads=NKV, sliding_window=WINDOW, rms_norm_eps=EPS,
        num_experts_per_tok=TOP_K, layer_types=["sliding_attention" if i % 2 == 0 else "full_attention"
                                                for i in range(LAYERS)],
        rope_parameters={"rope_type": "yarn", "rope_theta": THETA, "factor": FACTOR,
                         "original_max_position_embeddings": ORIG_CTX, "beta_fast": BETA_FAST,
                         "beta_slow": BETA_SLOW, "truncate": False},
        max_position_embeddings=131072, tie_word_embeddings=False, attention_bias=True)
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"  # its loop over the experts (grouped_mm takes no float64)
    model = GptOssForCausalLM(cfg).eval()
    sd = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    def rnd(shape, std):
        return rng.normal(0.0, std, size=shape).astype(np.float32)

    tensors = []  # (gguf name, data, raw dtype or None)

    def put(name, a, typ=None, hf=None, back=None):
        """a into the file (typ: GGML blocks, None: float32), and what the file holds into Hugging Face's tensor hf
        (back: from the file's layout to hf's)."""
        if typ is None:
            held = a.astype(np.float32)
            tensors.append((name, held, None))
        else:
            q, held = quantized(a, typ)
            tensors.append((name, q, typ))
        if hf is not None:
            v = back(held) if back else held
            assert sd[hf].shape == tuple(v.shape), (hf, sd[hf].shape, v.shape)
            sd[hf] = torch.from_numpy(np.ascontiguousarray(v))

    Q8, FP4 = (gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.MXFP4) if quant else (None, None)
    put("token_embd.weight", rnd((VOCAB, H), 1.0), Q8, "model.embed_tokens.weight")
    put("output_norm.weight", 1 + rnd((H,), 0.1), None, "model.norm.weight")
    put("output.weight", rnd((VOCAB, H), 4 / np.sqrt(H)), Q8, "lm_head.weight")  # logits far apart
    for i in range(LAYERS):
        b, p = f"blk.{i}.", f"model.layers.{i}."
        put(b + "attn_norm.weight", 1 + rnd((H,), 0.1), None, p + "input_layernorm.weight")
        put(b + "post_attention_norm.weight", 1 + rnd((H,), 0.1), None, p + "post_attention_layernorm.weight")
        for n, rows in (("q", NH * HD), ("k", NKV * HD), ("v", NKV * HD)):
            put(b + f"attn_{n}.weight", rnd((rows, H), 1 / np.sqrt(H)), Q8, p + f"self_attn.{n}_proj.weight")
            put(b + f"attn_{n}.bias", rnd((rows,), 0.2), None, p + f"self_attn.{n}_proj.bias")
        put(b + "attn_output.weight", rnd((H, NH * HD), 1 / np.sqrt(NH * HD)), Q8, p + "self_attn.o_proj.weight")
        put(b + "attn_output.bias", rnd((H,), 0.2), None, p + "self_attn.o_proj.bias")
        put(b + "attn_sinks.weight", rnd((NH,), 1.0), None, p + "self_attn.sinks")
        put(b + "ffn_gate_inp.weight", rnd((E, H), 8 / np.sqrt(H)), None, p + "mlp.router.weight")  # experts apart
        put(b + "ffn_gate_inp.bias", rnd((E,), 0.5), None, p + "mlp.router.bias")
        # Hugging Face's experts: gate_up [E, H, 2 I] with a gate column then its up column, down [E, I, H]
        gate, up = rnd((E, INTER, H), 1 / np.sqrt(H)), rnd((E, INTER, H), 1 / np.sqrt(H))
        down = rnd((E, H, INTER), 1 / np.sqrt(INTER))
        _, gate_h = quantized(gate, FP4)
        _, up_h = quantized(up, FP4)
        put(b + "ffn_gate_exps.weight", gate, FP4)
        put(b + "ffn_up_exps.weight", up, FP4)
        gu = np.empty((E, H, 2 * INTER), np.float32)
        gu[:, :, 0::2] = gate_h.transpose(0, 2, 1)
        gu[:, :, 1::2] = up_h.transpose(0, 2, 1)
        sd[p + "mlp.experts.gate_up_proj"] = torch.from_numpy(gu)
        put(b + "ffn_down_exps.weight", down, FP4, p + "mlp.experts.down_proj", lambda d: d.transpose(0, 2, 1))
        gb, ub = rnd((E, INTER), 0.2), rnd((E, INTER), 0.2)
        put(b + "ffn_gate_exps.bias", gb)
        put(b + "ffn_up_exps.bias", ub)
        gub = np.empty((E, 2 * INTER), np.float32)
        gub[:, 0::2], gub[:, 1::2] = gb, ub
        sd[p + "mlp.experts.gate_up_proj_bias"] = torch.from_numpy(gub)
        put(b + "ffn_down_exps.bias", rnd((E, H), 0.2), None, p + "mlp.experts.down_proj_bias")
    model.load_state_dict(sd)
    model = model.double()

    w = gguf.GGUFWriter(str(out / f"tiny_{stem}.gguf"), "gpt-oss")
    w.add_name("tiny-gpt-oss")
    w.add_quantization_version(gguf.GGML_QUANT_VERSION)
    w.add_block_count(LAYERS)
    w.add_context_length(131072)
    w.add_embedding_length(H)
    w.add_feed_forward_length(INTER)
    w.add_head_count(NH)
    w.add_head_count_kv(NKV)
    w.add_key_length(HD)
    w.add_value_length(HD)
    w.add_layer_norm_rms_eps(EPS)
    w.add_expert_count(E)
    w.add_expert_used_count(TOP_K)
    w.add_uint32("gpt-oss.expert_feed_forward_length", INTER)
    w.add_sliding_window(WINDOW)
    w.add_rope_freq_base(THETA)
    w.add_string("gpt-oss.rope.scaling.type", "yarn")
    w.add_float32("gpt-oss.rope.scaling.factor", FACTOR)
    w.add_uint32("gpt-oss.rope.scaling.original_context_length", ORIG_CTX)
    w.add_float32("gpt-oss.rope.scaling.yarn_beta_fast", BETA_FAST)
    w.add_float32("gpt-oss.rope.scaling.yarn_beta_slow", BETA_SLOW)
    b = byte_tokens()
    merged = [b[ord("t")] + b[ord("h")], b[ord("h")] + b[ord("e")]]  # two merges: a tokenizer has some
    tokens = b + SPECIALS + merged + [f"<|reserved_{i}|>" for i in range(VOCAB - 256 - len(SPECIALS) - len(merged))]
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "gpt-4o")
    w.add_array("tokenizer.ggml.tokens", tokens)
    w.add_array("tokenizer.ggml.token_type", [1] * 256 + [3] * len(SPECIALS) + [1] * len(merged) +
                [3] * (VOCAB - 256 - len(SPECIALS) - len(merged)))
    w.add_array("tokenizer.ggml.merges", [f"{b[ord('t')]} {b[ord('h')]}", f"{b[ord('h')]} {b[ord('e')]}"])
    special = {t: 256 + i for i, t in enumerate(SPECIALS)}
    w.add_uint32("tokenizer.ggml.bos_token_id", special["<|startoftext|>"])
    w.add_uint32("tokenizer.ggml.eos_token_id", special["<|return|>"])
    w.add_uint32("tokenizer.ggml.padding_token_id", special["<|endoftext|>"])
    for name, data, typ in tensors:
        if typ is None:
            w.add_tensor(name, data)
        else:
            w.add_tensor(name, data, raw_dtype=typ)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    eos = [special["<|return|>"], special["<|endoftext|>"], special["<|call|>"]]
    cases = []
    gaps = []  # the routers' smallest k-th to (k+1)-th gap in the last forward
    def router_gap(_module, _inputs, output):
        top = torch.topk(output[0], TOP_K + 1, dim=-1).values  # the router's logits (router_logits, scores, idx)
        gaps.append(float((top[..., TOP_K - 1] - top[..., TOP_K]).min()))
    for layer in model.model.layers:
        layer.mlp.router.register_forward_hook(router_gap)
    with torch.no_grad():
        for n in [int(x) for x in rng.integers(3, 41, size=600)]:
            if len(cases) == CASES:
                break
            prompt = [int(t) for t in rng.integers(0, 256, size=n)]
            ids, margin, gap = list(prompt), float("inf"), float("inf")
            for _ in range(NEW_TOKENS):
                gaps.clear()
                logits = model(torch.tensor([ids])).logits[0, -1]
                gap = min([gap] + gaps)
                top = torch.topk(logits, 2).values
                margin = min(margin, float(top[0] - top[1]))
                t = int(torch.argmax(logits))
                ids.append(t)
                if t in eos:
                    break
            if margin >= MIN_MARGIN and gap >= MIN_ROUTER_GAP:
                cases.append({"prompt": prompt, "tokens": ids[n:], "min_margin": margin, "min_router_gap": gap})
    assert len(cases) == CASES, f"only {len(cases)} cases with a clear margin"
    (out / f"golden_{stem}.json").write_text(json.dumps({"eos": eos, "cases": cases}), encoding="utf-8")
    stamp.write_text(inputs_digest())
