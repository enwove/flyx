# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""A tiny Gemma 4 as llama.cpp's converter lays it out (arch "gemma4"), and Hugging Face's own outputs for it.

    tiny_gemma4.gguf     6 layers (sliding, sliding, global, sliding, sliding, global; a window of 8): 4 query heads
                         of 32 over 2 K/V heads on the sliding layers, of 64 over 1 K/V head that is also v (k's
                         projection, attention_k_eq_v) on the global ones, whose proportional rotary turns 16 of the
                         64 dims; a hidden size of 64, GELU (tanh) MLPs of 96, RMSNorms before and after attention and
                         MLP, a learned scalar on each layer's output, the embedding (times sqrt(64)) tied to the head,
                         logits softcapped at 30.  Attention, MLP and the embedding in Q8_0, norms and scalars in
                         float32
    golden_gemma4.json   transformers' Gemma4ForCausalLM in float64 on the weights the file holds (every block
                         decoded back into it), greedy: each case's tokens and the smallest margin between the best
                         two logits along them; the cases kept have one of at least MIN_MARGIN and the same tokens
                         with every linear's weights and input rounded to bf16 (an engine's rounding then moves no
                         token); and each case's first drafts from its assistant (below), transformers' own
    tiny_gemma4_assistant/
                         its assistant (Gemma4AssistantForCausalLM: config.json, model.safetensors in bf16, as Google
                         ships them): a sliding and a global layer of 32 reading the tiny model's K/V (its last
                         sliding layer's and its last global one's), MLPs of 48, its own head; "drafts" in
                         golden_gemma4.json are what it drafts after each prompt, up to the first whose best two
                         logits are less than MIN_DRAFT_MARGIN apart

Its vocabulary: the 256 bytes as GPT-2 writes them, a few special tokens, reserved ones up to 512 (the model, not
Gemma's tokenizer, is what these files test).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from importlib import metadata
from pathlib import Path

import gguf
import numpy as np

H, NH, INTER, WINDOW, VOCAB = 64, 4, 96, 8, 512
SLIDING = [True, True, False, True, True, False]
NKV_SWA, HD_SWA, NKV_FULL, HD_FULL, ROTARY = 2, 32, 1, 64, 0.25
THETA, THETA_SWA, EPS, SOFTCAP = 1000000.0, 10000.0, 1e-6, 30.0
SPECIALS = ["<pad>", "<eos>", "<bos>", "<unk>", "<turn|>"]
MIN_MARGIN, NEW_TOKENS, CASES = 0.15, 12, 8
A_H, A_INTER, A_LAYERS, DRAFTS, MIN_DRAFT_MARGIN = 32, 48, [True, False], 7, 0.25  # the assistant
QK_LO, QK_HI = 0.6, 0.8  # q and k norm weights (see put below)


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


def write(out: Path, quant: bool = True, stem: str = "gemma4"):
    """quant False: every tensor float32 (no input rounding anywhere).  Skipped where the files are there, made from
    the same inputs (their stamp)."""
    stamp = out / f"tiny_{stem}.stamp"
    assistant_dir = out / f"tiny_{stem}_assistant"
    if stamp.exists() and stamp.read_text() == inputs_digest() and (out / f"tiny_{stem}.gguf").exists() and \
            (out / f"golden_{stem}.json").exists() and (not quant or (assistant_dir / "model.safetensors").exists()):
        return
    import torch
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    torch.set_num_threads(min(4, os.cpu_count() or 1))  # a tiny model's products: more threads only spin
    torch.manual_seed(20261128)
    rng = np.random.default_rng(20261128)
    layers = len(SLIDING)
    cfg = Gemma4TextConfig(
        vocab_size=VOCAB, hidden_size=H, intermediate_size=INTER, num_hidden_layers=layers, num_attention_heads=NH,
        num_key_value_heads=NKV_SWA, head_dim=HD_SWA, global_head_dim=HD_FULL, num_global_key_value_heads=NKV_FULL,
        sliding_window=WINDOW, layer_types=["sliding_attention" if s else "full_attention" for s in SLIDING],
        rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": THETA_SWA},
                         "full_attention": {"rope_type": "proportional", "partial_rotary_factor": ROTARY,
                                            "rope_theta": THETA}},
        final_logit_softcapping=SOFTCAP, rms_norm_eps=EPS, hidden_size_per_layer_input=0, attention_k_eq_v=True,
        num_kv_shared_layers=0, enable_moe_block=False, tie_word_embeddings=True, max_position_embeddings=131072,
        pad_token_id=0, eos_token_id=1, bos_token_id=2)
    cfg._attn_implementation = "eager"
    model = Gemma4ForCausalLM(cfg).eval()
    sd = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    def rnd(shape, std):
        return rng.normal(0.0, std, size=shape).astype(np.float32)

    tensors = []  # (gguf name, data, raw dtype or None)

    def put(name, a, typ=None, hf=None):
        """a into the file (typ: GGML blocks, None: float32), and what the file holds into Hugging Face's tensor hf."""
        if typ is None:
            held = a.astype(np.float32)
            tensors.append((name, held, None))
        else:
            q, held = quantized(a, typ)
            tensors.append((name, q, typ))
        if hf is not None:
            assert sd[hf].shape == tuple(held.shape), (hf, sd[hf].shape, held.shape)
            sd[hf] = torch.from_numpy(np.ascontiguousarray(held))

    Q8 = gguf.GGMLQuantizationType.Q8_0 if quant else None
    # the embedding times sqrt(64) below the layers' normed outputs (their post-norms' weights about 2: the head,
    # tied to the embedding, would otherwise give back the token it read); the final norm's weights spread the
    # head's logits
    put("token_embd.weight", rnd((VOCAB, H), 0.25), Q8, "model.embed_tokens.weight")
    put("output_norm.weight", 3 + rnd((H,), 0.3), None, "model.norm.weight")
    rot = int(ROTARY * HD_FULL // 2)  # the global layers' turning pairs; llama.cpp's factors keep the others still
    tensors.append(("rope_freqs.weight", np.array([1.0] * rot + [1e30] * (HD_FULL // 2 - rot), np.float32), None))
    for i, sliding in enumerate(SLIDING):
        b, p = f"blk.{i}.", f"model.layers.{i}."
        nkv, hd = (NKV_SWA, HD_SWA) if sliding else (NKV_FULL, HD_FULL)
        put(b + "attn_norm.weight", 1 + rnd((H,), 0.1), None, p + "input_layernorm.weight")
        put(b + "attn_q.weight", rnd((NH * hd, H), 1 / np.sqrt(H)), Q8, p + "self_attn.q_proj.weight")
        put(b + "attn_k.weight", rnd((nkv * hd, H), 1 / np.sqrt(H)), Q8, p + "self_attn.k_proj.weight")
        if sliding:  # the global layers' v is k's projection
            put(b + "attn_v.weight", rnd((nkv * hd, H), 1 / np.sqrt(H)), Q8, p + "self_attn.v_proj.weight")
        # q.k is unscaled over q and k normed per head: norm weights of 1 would make each head's softmax nearly an
        # argmax, where any rounding of an input may move which key it picks; these keep its logits a few apart
        put(b + "attn_q_norm.weight", rng.uniform(QK_LO, QK_HI, size=(hd,)).astype(np.float32), None,
            p + "self_attn.q_norm.weight")
        put(b + "attn_k_norm.weight", rng.uniform(QK_LO, QK_HI, size=(hd,)).astype(np.float32), None,
            p + "self_attn.k_norm.weight")
        put(b + "attn_output.weight", rnd((H, NH * hd), 1 / np.sqrt(NH * hd)), Q8, p + "self_attn.o_proj.weight")
        put(b + "post_attention_norm.weight", 2 + rnd((H,), 0.2), None, p + "post_attention_layernorm.weight")
        put(b + "ffn_norm.weight", 1 + rnd((H,), 0.1), None, p + "pre_feedforward_layernorm.weight")
        put(b + "ffn_gate.weight", rnd((INTER, H), 1 / np.sqrt(H)), Q8, p + "mlp.gate_proj.weight")
        put(b + "ffn_up.weight", rnd((INTER, H), 1 / np.sqrt(H)), Q8, p + "mlp.up_proj.weight")
        put(b + "ffn_down.weight", rnd((H, INTER), 1 / np.sqrt(INTER)), Q8, p + "mlp.down_proj.weight")
        put(b + "post_ffw_norm.weight", 2 + rnd((H,), 0.2), None, p + "post_feedforward_layernorm.weight")
        put(b + "layer_output_scale.weight", rng.uniform(0.5, 1.5, size=(1,)).astype(np.float32), None,
            p + "layer_scalar")
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"]  # tied: the state's own copy would load over the table
    model.load_state_dict(sd)
    model = model.double()

    w = gguf.GGUFWriter(str(out / f"tiny_{stem}.gguf"), "gemma4")
    w.add_name("tiny-gemma4")
    w.add_quantization_version(gguf.GGML_QUANT_VERSION)
    a = "gemma4."
    w.add_uint32(a + "block_count", layers)
    w.add_uint32(a + "context_length", 131072)
    w.add_uint32(a + "embedding_length", H)
    w.add_uint32(a + "feed_forward_length", INTER)
    w.add_uint32(a + "attention.head_count", NH)
    w.add_array(a + "attention.head_count_kv", [NKV_SWA if s else NKV_FULL for s in SLIDING])
    w.add_float32(a + "rope.freq_base", THETA)
    w.add_float32(a + "rope.freq_base_swa", THETA_SWA)
    w.add_float32(a + "attention.layer_norm_rms_epsilon", EPS)
    w.add_uint32(a + "attention.key_length", HD_FULL)
    w.add_uint32(a + "attention.value_length", HD_FULL)
    w.add_float32(a + "final_logit_softcapping", SOFTCAP)
    w.add_uint32(a + "attention.sliding_window", WINDOW)
    w.add_uint32(a + "attention.shared_kv_layers", 0)
    w.add_uint32(a + "embedding_length_per_layer_input", 0)
    w.add_array(a + "attention.sliding_window_pattern", list(SLIDING))
    w.add_uint32(a + "attention.key_length_swa", HD_SWA)
    w.add_uint32(a + "attention.value_length_swa", HD_SWA)
    w.add_uint32(a + "rope.dimension_count", HD_FULL)
    w.add_uint32(a + "rope.dimension_count_swa", HD_SWA)
    bt = byte_tokens()
    merged = [bt[ord("t")] + bt[ord("h")], bt[ord("h")] + bt[ord("e")]]  # two merges: a tokenizer has some
    reserved = VOCAB - 256 - len(SPECIALS) - len(merged)
    tokens = bt + SPECIALS + merged + [f"<|reserved_{i}|>" for i in range(reserved)]
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "gpt-4o")
    w.add_array("tokenizer.ggml.tokens", tokens)
    w.add_array("tokenizer.ggml.token_type", [1] * 256 + [3] * len(SPECIALS) + [1] * len(merged) + [3] * reserved)
    w.add_array("tokenizer.ggml.merges", [f"{bt[ord('t')]} {bt[ord('h')]}", f"{bt[ord('h')]} {bt[ord('e')]}"])
    special = {t: 256 + i for i, t in enumerate(SPECIALS)}
    w.add_uint32("tokenizer.ggml.bos_token_id", special["<bos>"])
    w.add_uint32("tokenizer.ggml.eos_token_id", special["<turn|>"])
    w.add_uint32("tokenizer.ggml.padding_token_id", special["<pad>"])
    for name, data, typ in tensors:
        if typ is None:
            w.add_tensor(name, data)
        else:
            w.add_tensor(name, data, raw_dtype=typ)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    eos = [special["<turn|>"], special["<eos>"]]

    # The same model with every linear's weights and input rounded to bf16, the coarsest an engine reads one (a
    # prefill pass's bf16 tensor cores, which take a float32 file's matrices in bf16): a case kept gives the same
    # tokens under it too.  Its attention (q.k unscaled) may amplify such a rounding past what a margin on the next
    # token shows.
    coarse = copy.deepcopy(model)
    for mod in coarse.modules():
        if isinstance(mod, torch.nn.Linear):
            mod.weight.data = mod.weight.data.to(torch.bfloat16).to(mod.weight.dtype)
            mod.register_forward_pre_hook(lambda _m, inp: (inp[0].to(torch.bfloat16).to(inp[0].dtype),))

    def greedy(m, prompt):
        ids, margin = list(prompt), float("inf")
        for _ in range(NEW_TOKENS):
            logits = m(torch.tensor([ids])).logits[0, -1]
            top = torch.topk(logits, 2).values
            margin = min(margin, float(top[0] - top[1]))
            t = int(torch.argmax(logits))
            ids.append(t)
            if t in eos:
                break
        return ids[len(prompt):], margin

    cases = []
    with torch.no_grad():
        for n in [int(x) for x in rng.integers(3, 41, size=600)]:
            if len(cases) == CASES:
                break
            prompt = [int(t) for t in rng.integers(0, 256, size=n)]
            tokens, margin = greedy(model, prompt)
            if margin >= MIN_MARGIN and greedy(coarse, prompt)[0] == tokens:
                cases.append({"prompt": prompt, "tokens": tokens, "min_margin": margin})
    assert len(cases) == CASES, f"only {len(cases)} cases with a clear margin"
    if quant:
        write_assistant(assistant_dir, model, cases, rng)
    (out / f"golden_{stem}.json").write_text(json.dumps({"eos": eos, "cases": cases}), encoding="utf-8")
    stamp.write_text(inputs_digest())


if __name__ == "__main__":
    import sys
    write(Path(sys.argv[1]))
    write(Path(sys.argv[1]), quant=False, stem="gemma4_f32")


def write_assistant(out: Path, target, cases, rng):
    """The tiny model's assistant into out, and each case's drafts from it (case["drafts"]): its steps at the position
    after the prompt, each reading the target's K/V before it (a sliding layer's from WINDOW - 1 back), as flyx's
    drafts read them (backend_abi.h flyx_gemma4_mtp_block)."""
    import torch
    from safetensors.torch import save_file
    from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM, Gemma4TextConfig

    tcfg = Gemma4TextConfig(
        vocab_size=VOCAB, hidden_size=A_H, intermediate_size=A_INTER, num_hidden_layers=len(A_LAYERS),
        num_attention_heads=NH, num_key_value_heads=NKV_SWA, head_dim=HD_SWA, global_head_dim=HD_FULL,
        num_global_key_value_heads=NKV_FULL, sliding_window=WINDOW,
        layer_types=["sliding_attention" if s else "full_attention" for s in A_LAYERS],
        rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": THETA_SWA},
                         "full_attention": {"rope_type": "proportional", "partial_rotary_factor": ROTARY,
                                            "rope_theta": THETA}},
        final_logit_softcapping=None, rms_norm_eps=EPS, hidden_size_per_layer_input=0, vocab_size_per_layer_input=0,
        attention_k_eq_v=True,
        num_kv_shared_layers=len(A_LAYERS), enable_moe_block=False, tie_word_embeddings=True,
        max_position_embeddings=131072, pad_token_id=0, eos_token_id=1, bos_token_id=2)
    cfg = Gemma4AssistantConfig(text_config=tcfg, backbone_hidden_size=H, use_ordered_embeddings=False)
    cfg.architectures = ["Gemma4AssistantForCausalLM"]
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    model = Gemma4AssistantForCausalLM(cfg).eval()

    def rnd(shape, std):
        return torch.from_numpy(rng.normal(0.0, std, size=shape).astype(np.float32))

    sd = {}
    shapes = {k: tuple(v.shape) for k, v in model.state_dict().items() if k != "lm_head.weight"}
    for k, shape in shapes.items():
        if k.endswith("layer_scalar"):
            v = torch.from_numpy(rng.uniform(0.5, 1.5, size=shape).astype(np.float32))
        elif k.endswith("q_norm.weight"):
            v = torch.from_numpy(rng.uniform(QK_LO, QK_HI, size=shape).astype(np.float32))
        elif k.endswith("post_attention_layernorm.weight") or k.endswith("post_feedforward_layernorm.weight"):
            v = 2 + rnd(shape, 0.2)
        elif k == "model.norm.weight":
            v = 3 + rnd(shape, 0.3)
        elif k.endswith("norm.weight"):
            v = 1 + rnd(shape, 0.1)
        elif k == "model.embed_tokens.weight":
            v = rnd(shape, 0.25)
        else:  # a linear [out, in]
            v = rnd(shape, 1 / np.sqrt(shape[1]))
        sd[k] = v.to(torch.bfloat16)  # as the checkpoints hold them
    out.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    cfg.to_json_file(str(out / "config.json"), use_diff=False)  # every field, as the checkpoints list them
    full = {k: v.double() for k, v in sd.items()}
    full["lm_head.weight"] = full["model.embed_tokens.weight"]
    model.load_state_dict(full)
    model = model.double()

    with torch.no_grad():
        for case in cases:
            prompt = case["prompt"]
            ids = torch.tensor([prompt])
            base = target.model(input_ids=ids, return_shared_kv_states=True, use_cache=False)
            kv = base.shared_kv_states
            h = base.last_hidden_state[:, -1:]  # the target's final-normed hidden at the prompt's last position
            tok = int(torch.argmax(target(input_ids=ids, use_cache=False).logits[0, -1]))  # the anchor
            p = len(prompt)
            keys = torch.arange(p)
            full_mask = torch.zeros(1, 1, 1, p, dtype=torch.float64)
            swa_mask = torch.where(keys >= p - WINDOW + 1, 0.0, float("-inf")).to(torch.float64).view(1, 1, 1, p)
            drafts = []
            for _ in range(DRAFTS):
                e = target.model.embed_tokens(torch.tensor([[tok]]))
                x = model.pre_projection(torch.cat([e, h], dim=-1))
                o = model.model(inputs_embeds=x, position_ids=torch.tensor([[p]]), shared_kv_states=kv,
                                attention_mask={"full_attention": full_mask, "sliding_attention": swa_mask},
                                use_cache=False)
                g = o.last_hidden_state
                logits = model.lm_head(g)[0, -1]
                top = torch.topk(logits, 2).values
                if float(top[0] - top[1]) < MIN_DRAFT_MARGIN:
                    break
                tok = int(torch.argmax(logits))
                drafts.append(tok)
                h = model.post_projection(g)
            case["drafts"] = drafts
