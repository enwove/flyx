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
    tiny_gemma4_assistant.gguf
                         the same assistant as llama.cpp's converter lays it out (arch "gemma4-assistant"), in
                         float32
    hf_gemma4/, tiny_gemma4.flyx
                         the tiny model as a Hugging Face directory (Gemma4ForCausalLM: config.json,
                         model.safetensors in float32: the values the GGUF file's blocks decode to; tokenizer.json:
                         its vocabulary and merges as a byte-level BPE), and python/convert's .flyx of it
    tiny_gemma4_moe.gguf, golden_gemma4_moe.json
                         the same layers with experts beside every MLP as 26B-A4B's (8 of 32 wide, the 2 best of a
                         row; the router over the layer's input normed without a weight, times its scale and
                         1 / sqrt(64); every expert's scale; the dense MLP's and the experts' outputs each normed, their
                         sum normed again), in Q8_0, and transformers' tokens for it
    hf_gemma4_nvfp4/, tiny_gemma4_nvfp4.flyx
                         the same with its MLPs' linears in NVFP4 as ModelOpt writes them (its attention left in
                         float, as NVIDIA's Gemma-4-31B-IT-NVFP4 leaves it), and its .flyx

Its vocabulary: the 256 bytes as GPT-2 writes them, a few special tokens, reserved ones up to 512 (the model, not
Gemma's tokenizer, is what these files test).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
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
N_EXPERTS, TOP_K, MOE_INTER = 8, 2, 32  # the experts (stem gemma4_moe)
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


def write_hf(out: Path, cfg, sd, tokens, merges, specials):
    """The model as transformers saves it (its tied head left out), with a byte-level BPE tokenizer.json of its
    vocabulary."""
    import torch
    from safetensors.torch import save_file
    from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers
    from flyx_convert.tokenizer_export import QWEN35_SPLIT
    out.mkdir(parents=True, exist_ok=True)
    c = copy.deepcopy(cfg)
    c.architectures = ["Gemma4ForCausalLM"]
    c.to_json_file(str(out / "config.json"), use_diff=False)
    save_file({k: v.to(torch.float32).contiguous() for k, v in sd.items() if k != "lm_head.weight"},
              str(out / "model.safetensors"), metadata={"format": "pt"})
    tok = Tokenizer(models.BPE(vocab={t: i for i, t in enumerate(tokens)}, merges=merges))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split(Regex(QWEN35_SPLIT), behavior="isolated", invert=False),
         pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(specials)
    tok.save(str(out / "tokenizer.json"))


def write_nvfp4(src: Path, dst: Path):
    """src with its MLPs' linears in ModelOpt's NVFP4 (tests/e2e/make_tiny.py's quantize_nvfp4), the attention's left
    as they are and named in the config's ignore list, as NVIDIA's Gemma 4 checkpoints."""
    import shutil
    from safetensors.torch import load_file, save_file
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "e2e"))
    from make_tiny import quantize_nvfp4  # noqa: E402
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "tokenizer.json", dst / "tokenizer.json")
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    cfg["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4",
                                  "producer": {"name": "flyx tiny_gemma4"},
                                  "ignore": ["lm_head"] + [f"model.layers.{i}.self_attn*" for i in range(len(SLIDING))]}
    (dst / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    import torch
    out = {}
    for k, v in load_file(str(src / "model.safetensors")).items():
        base = k[: -len(".weight")]
        if k.endswith(".weight") and any(base.endswith(m) for m in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")):
            q, sc, s2 = quantize_nvfp4(v)
            out[k], out[base + ".weight_scale"], out[base + ".weight_scale_2"] = q, sc, s2
            out[base + ".input_scale"] = torch.tensor(1.0)
        else:
            out[k] = v
    save_file({k: t.contiguous() for k, t in out.items()}, str(dst / "model.safetensors"))


def write(out: Path, quant: bool = True, stem: str = "gemma4", moe: bool = False):
    """quant False: every tensor float32 (no input rounding anywhere); moe: experts beside every MLP.  Skipped where
    the files are there, made from the same inputs (their stamp)."""
    extras = quant and not moe  # the assistant, the HF directory and its .flyx files: the dense Q8 model's
    stamp = out / f"tiny_{stem}.stamp"
    assistant_dir = out / f"tiny_{stem}_assistant"
    made = [out / f"tiny_{stem}.gguf", out / f"golden_{stem}.json"]
    if extras:
        made += [assistant_dir / "model.safetensors", out / f"tiny_{stem}_assistant.gguf", out / f"tiny_{stem}.flyx",
                 out / f"tiny_{stem}_nvfp4.flyx"]
    if stamp.exists() and stamp.read_text() == inputs_digest() and all(m.exists() for m in made):
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
        num_kv_shared_layers=0, enable_moe_block=moe, tie_word_embeddings=True, max_position_embeddings=131072,
        pad_token_id=0, eos_token_id=1, bos_token_id=2,
        **({"num_experts": N_EXPERTS, "top_k_experts": TOP_K, "moe_intermediate_size": MOE_INTER} if moe else {}))
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"  # each expert's own products (the grouped ones take no float64)
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
        if moe:  # the experts beside the MLP, and the norms around both (llama.cpp's names)
            E, MI = N_EXPERTS, MOE_INTER
            put(b + "post_ffw_norm_1.weight", 1 + rnd((H,), 0.1), None, p + "post_feedforward_layernorm_1.weight")
            put(b + "pre_ffw_norm_2.weight", 1 + rnd((H,), 0.1), None, p + "pre_feedforward_layernorm_2.weight")
            put(b + "post_ffw_norm_2.weight", 1 + rnd((H,), 0.1), None, p + "post_feedforward_layernorm_2.weight")
            # the router's logits apart (its input normed and times 1 / sqrt(H): rows of 32 / sqrt(H) give logits of
            # about four), so a rounding of an input moves few rows' choice of experts
            put(b + "ffn_gate_inp.weight", rnd((E, H), 32.0 / np.sqrt(H)), None, p + "router.proj.weight")
            put(b + "ffn_gate_inp.scale", rng.uniform(0.8, 1.2, size=(H,)).astype(np.float32), None,
                p + "router.scale")
            put(b + "ffn_down_exps.scale", rng.uniform(0.5, 1.5, size=(E,)).astype(np.float32), None,
                p + "router.per_expert_scale")
            put(b + "ffn_gate_up_exps.weight", rnd((E, 2 * MI, H), 1 / np.sqrt(H)), Q8, p + "experts.gate_up_proj")
            put(b + "ffn_down_exps.weight", rnd((E, H, MI), 1 / np.sqrt(MI)), Q8, p + "experts.down_proj")
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
    if moe:
        w.add_uint32(a + "expert_count", N_EXPERTS)
        w.add_uint32(a + "expert_used_count", TOP_K)
        w.add_uint32(a + "expert_feed_forward_length", MOE_INTER)
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
    if extras:  # the same weights (the blocks' values) through transformers' layout and python/convert
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python" / "convert"))
        from flyx_convert import convert  # noqa: E402
        merge_pairs = [(bt[ord("t")], bt[ord("h")]), (bt[ord("h")], bt[ord("e")])]
        write_hf(out / f"hf_{stem}", cfg, sd, tokens, merge_pairs, SPECIALS)
        convert(str(out / f"hf_{stem}"), str(out / f"tiny_{stem}.flyx"), None, log=lambda *_: None)
        write_nvfp4(out / f"hf_{stem}", out / f"hf_{stem}_nvfp4")
        convert(str(out / f"hf_{stem}_nvfp4"), str(out / f"tiny_{stem}_nvfp4.flyx"), None, log=lambda *_: None)

    # The same model with every linear's weights and input rounded to bf16, the coarsest an engine reads one (a
    # prefill pass's bf16 tensor cores, which take a float32 file's matrices in bf16): a case kept gives the same
    # tokens under it too.  Its attention (q.k unscaled) may amplify such a rounding past what a margin on the next
    # token shows.
    coarse = copy.deepcopy(model)
    for mod in coarse.modules():
        if isinstance(mod, torch.nn.Linear):
            mod.weight.data = mod.weight.data.to(torch.bfloat16).to(mod.weight.dtype)
            mod.register_forward_pre_hook(lambda _m, inp: (inp[0].to(torch.bfloat16).to(inp[0].dtype),))
        elif type(mod).__name__ == "Gemma4TextExperts":  # the experts too (their routing then moves with no rounding)
            for w in (mod.gate_up_proj, mod.down_proj):
                w.data = w.data.to(torch.bfloat16).to(w.dtype)
            mod.register_forward_pre_hook(
                lambda _m, inp: (inp[0].to(torch.bfloat16).to(inp[0].dtype),) + tuple(inp[1:]))

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
    if extras:
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
    write_assistant_gguf(out.parent / (out.name + ".gguf"), {k: v.float().numpy() for k, v in sd.items()})
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


def write_assistant_gguf(path: Path, w):
    """The assistant's tensors w (transformers' names, float32) as llama.cpp's Gemma4AssistantModel writes them."""
    a = "gemma4-assistant."
    g = gguf.GGUFWriter(str(path), "gemma4-assistant")
    g.add_name("tiny-gemma4-assistant")
    g.add_uint32(a + "block_count", len(A_LAYERS))
    g.add_uint32(a + "context_length", 131072)
    g.add_uint32(a + "embedding_length", A_H)
    g.add_uint32(a + "feed_forward_length", A_INTER)
    g.add_uint32(a + "attention.head_count", NH)
    g.add_array(a + "attention.head_count_kv", [NKV_SWA if s else NKV_FULL for s in A_LAYERS])
    g.add_float32(a + "rope.freq_base", THETA)
    g.add_float32(a + "rope.freq_base_swa", THETA_SWA)
    g.add_float32(a + "attention.layer_norm_rms_epsilon", EPS)
    g.add_uint32(a + "attention.key_length", HD_FULL)
    g.add_uint32(a + "attention.value_length", HD_FULL)
    g.add_uint32(a + "attention.sliding_window", WINDOW)
    g.add_uint32(a + "attention.shared_kv_layers", len(A_LAYERS))
    g.add_uint32(a + "embedding_length_per_layer_input", 0)
    g.add_array(a + "attention.sliding_window_pattern", list(A_LAYERS))
    g.add_uint32(a + "attention.key_length_swa", HD_SWA)
    g.add_uint32(a + "attention.value_length_swa", HD_SWA)
    g.add_uint32(a + "rope.dimension_count", HD_FULL)
    g.add_uint32(a + "rope.dimension_count_swa", HD_SWA)
    g.add_uint32(a + "embedding_length_out", H)
    g.add_uint32(a + "nextn_predict_layers", len(A_LAYERS))
    rot = int(ROTARY * HD_FULL // 2)
    g.add_tensor("rope_freqs.weight", np.array([1.0] * rot + [1e30] * (HD_FULL // 2 - rot), np.float32))
    g.add_tensor("token_embd.weight", w["model.embed_tokens.weight"])
    names = {"input_layernorm": "attn_norm", "self_attn.q_proj": "attn_q", "self_attn.o_proj": "attn_output",
             "self_attn.q_norm": "attn_q_norm", "post_attention_layernorm": "post_attention_norm",
             "pre_feedforward_layernorm": "ffn_norm", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
             "mlp.down_proj": "ffn_down", "post_feedforward_layernorm": "post_ffw_norm"}
    for i in range(len(A_LAYERS)):
        p = f"model.layers.{i}."
        for hf, gg in names.items():
            g.add_tensor(f"blk.{i}.{gg}.weight", w[p + hf + ".weight"])
        g.add_tensor(f"blk.{i}.layer_output_scale.weight", w[p + "layer_scalar"])
    g.add_tensor("output_norm.weight", w["model.norm.weight"])
    g.add_tensor("nextn.pre_projection.weight", w["pre_projection.weight"])
    g.add_tensor("nextn.post_projection.weight", w["post_projection.weight"])
    g.write_header_to_file()
    g.write_kv_data_to_file()
    g.write_tensors_to_file()
    g.close()
