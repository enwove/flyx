# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Tiny text checkpoints with the Hugging Face eager model as an independent logits oracle."""

import json
import sys
from pathlib import Path

import gguf
import torch
from tokenizers.pre_tokenizers import ByteLevel
from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "reference"))
from flyx_ref.qwen4exp import logits as reference_logits, q8_inputs


def write_fixture(out, ple, mixed=False):
    torch.manual_seed(31415)
    H, C, D, I, R = 64, 4, 32, 64, 16
    tokens = sorted(ByteLevel.alphabet()) + ["aa", "<|im_end|>", "<|endoftext|>"]
    eos = len(tokens) - 1
    nk, nv = 2, 4
    cfg = Qwen4ExpTextConfig(
        vocab_size=len(tokens),
        hidden_size=H,
        hc_count=C,
        hc_lowrank=R,
        head_dim=D,
        num_hidden_layers=2,
        full_attention_interval=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        linear_key_head_dim=D,
        linear_value_head_dim=D,
        linear_num_key_heads=nk,
        linear_num_value_heads=nv,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=I,
        shared_expert_intermediate_size=I,
        max_position_embeddings=128,
        output_gate_type="sigmoid",
        eos_token_id=eos,
        rope_parameters=dict(
            rope_type="default",
            rope_theta=10000.0,
            partial_rotary_factor=0.5,
            mrope_section=[3, 3, 2],
            mrope_interleaved=True,
        ),
        ple_layer_ids=[1] if ple else [],
        ple_embed_dim=H,
        heads_per_ngram=2,
        ngram_vocab_size_base=97,
        make_ngram_vocab_size_divisible_by=32,
        split_ngram_parts=1,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=D,
        indexer_compress_ratio=4,
        indexer_budget=8 if ple else 128,
    )
    cfg._attn_implementation = "eager"
    model = Qwen4ExpForCausalLM(cfg).cpu().eval()
    with torch.no_grad():
        for name, v in model.named_parameters():
            if v.ndim > 1:
                v.normal_(0.0, v.shape[-1] ** -0.5)
            elif name.endswith("norm.weight") and ".linear_attn." not in name:
                v.normal_(0.0, 0.13)
            v.copy_(v.bfloat16().float())
        # ReLU makes exact zero-score ties common for two random query heads. Opposite
        # heads give distinct absolute scores for the cross-platform Transformers oracle.
        # The dedicated selection test exercises the engine's chronological tie policy.
        index_qk = model.model.layers[1].self_attn.indexer.index_qk_proj.weight
        index_qk[D : 2 * D].copy_(-index_qk[:D])
    stem = "qwen4exp_mixed" if mixed else "qwen4exp_parent" if ple else "qwen4exp_coder"
    w = gguf.GGUFWriter(out / (stem + ".gguf"), "qwen4exp")
    w.add_quantization_version(2)
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", "qwen35")
    w.add_array("tokenizer.ggml.tokens", tokens)
    w.add_array("tokenizer.ggml.token_type", [1] * (len(tokens) - 2) + [3, 3])
    w.add_array("tokenizer.ggml.merges", ["a a"])
    w.add_uint32("tokenizer.ggml.eos_token_id", eos - 1)
    w.add_uint32("tokenizer.ggml.bos_token_id", eos)

    def integer(k, v):
        w.add_uint32("qwen4exp." + k, v)

    def array(k, v):
        w.add_key_value("qwen4exp." + k, v, gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT64)

    for k, v in {
        "embedding_length": H,
        "context_length": 128,
        "block_count": 2,
        "full_attention_interval": 2,
        "hyper_connection.count": C,
        "hyper_connection.low_rank": R,
        "ssm.group_count": nk,
        "ssm.time_step_rank": nv,
        "ssm.state_size": D,
        "ssm.inner_size": nv * D,
        "ssm.conv_kernel": 4,
        "attention.head_count": 2,
        "attention.head_count_kv": 1,
        "attention.key_length": D,
        "attention.value_length": D,
        "rope.dimension_count": D // 2,
        "attention.indexer.head_count": 2,
        "attention.indexer.key_length": D,
        "attention.indexer.top_k": 8,
        "expert_count": 4,
        "expert_used_count": 2,
        "expert_feed_forward_length": I,
        "expert_shared_feed_forward_length": I,
    }.items():
        integer(k, v)
    w.add_float32("qwen4exp.rope.freq_base", 10000.0)
    array("attention.compress_ratios", [0, 4 if ple else 0])
    array("rope.dimension_sections", [3, 3, 2, 0])

    def tensor(name, v):
        w.add_tensor(name, v.detach().float().contiguous().numpy())

    def quantized(name, value, dtype):
        raw = gguf.quants.quantize(value.detach().contiguous().numpy(), dtype)
        w.add_tensor(name, raw, raw_dtype=dtype)
        with torch.no_grad():
            value.copy_(torch.from_numpy(gguf.quants.dequantize(raw, dtype)))

    def hc(prefix, block):
        tensor(prefix + "norm.weight", 1 + block.hc_norm.weight)
        tensor(prefix + "down.weight", block.input_mix_weight_down.weight)
        tensor(prefix + "up.weight", block.input_mix_weight_up.weight)
        if block.block_inject_weight is not None:
            tensor(prefix + "inject.weight", block.block_inject_weight.weight)

    tensor("token_embd.weight", model.model.embed_tokens.weight)
    tensor("output.weight", model.lm_head.weight)
    hc("output_hc_", model.model.hyper_connection_mixer)
    for i, layer in enumerate(model.model.layers):
        p = f"blk.{i}."
        hc(p + "hc_attn_", layer.attn_hyper_connection)
        hc(p + "hc_ffn_", layer.mlp_hyper_connection)
        f = layer.mlp
        tensor(p + "ffn_gate_inp.weight", f.gate.weight)
        gate, up = f.experts.gate_up_proj.chunk(2, dim=1)
        if mixed and i == 0:
            quantized(p + "ffn_gate_exps.weight", gate, gguf.GGMLQuantizationType.Q8_0)
            quantized(p + "ffn_up_exps.weight", up, gguf.GGMLQuantizationType.Q4_0)
        elif not ple and i == 1:
            tensor(p + "ffn_gate_up_exps.weight", f.experts.gate_up_proj)
        else:
            tensor(p + "ffn_gate_exps.weight", gate)
            tensor(p + "ffn_up_exps.weight", up)
        tensor(p + "ffn_down_exps.weight", f.experts.down_proj)
        for key in ("gate", "up", "down"):
            tensor(p + "ffn_" + key + "_shexp.weight", getattr(f.shared_expert, key + "_proj").weight)
        tensor(p + "ffn_gate_inp_shexp.weight", f.shared_expert_gate.weight.flatten())
        if i == 0:
            g = layer.linear_attn
            # GGUF tiles value heads across key heads, while Transformers groups them by key.
            heads = torch.arange(nv).reshape(nk, nv // nk).T.flatten()
            features = (heads[:, None] * D + torch.arange(D)).flatten()
            qk = torch.arange(2 * nk * D)
            channels = torch.cat((qk, qk.numel() + features))
            for k, v in {
                "attn_qkv.weight": g.in_proj_qkv.weight[channels],
                "attn_gate.weight": g.in_proj_z.weight[features],
                "ssm_beta.weight": g.in_proj_b.weight[heads],
                "ssm_alpha.weight": g.in_proj_a.weight[heads],
                "ssm_out.weight": g.out_proj.weight[:, features],
                "ssm_conv1d.weight": g.conv1d.weight[channels, 0],
                "ssm_a": -g.A_log[heads].exp(),
                "ssm_norm.weight": g.norm.weight,
                "ssm_dt.bias": g.dt_bias[heads],
            }.items():
                tensor(p + k, v)
        else:
            a = layer.self_attn
            for k in ("q", "k", "v", "o"):
                tensor(p + "attn_" + ("output" if k == "o" else k) + ".weight", getattr(a, k + "_proj").weight)
            for k in ("q", "k"):
                tensor(p + "attn_" + k + "_norm.weight", 1 + getattr(a, k + "_norm").weight)
                tensor(p + "indexer." + k + "_norm.weight", 1 + getattr(a.indexer, k + "_layernorm").weight)
            q, k = a.indexer.index_qk_proj.weight.split([2 * D, D])
            tensor(p + "indexer.q_proj.weight", q)
            tensor(p + "indexer.k_proj.weight", k)
        if layer.ple is not None:
            b = layer.ple
            e = b.ple_embedding
            for key, v in {"ngram_size": 3, "heads_per_ngram": 2, "conv_kernel": 4, "eos_token_id": eos}.items():
                integer("ple." + key, v)
            integer("embedding_length_per_layer_input", H // 4)
            array("ple.layers", [i])
            for k, v in {
                "layer_multipliers": e.layer_multipliers,
                "head_vocab_sizes": e.ngram_heads_vocab_sizes,
                "head_offsets": e.ngram_heads_offsets,
            }.items():
                array("ple." + k, v.tolist())
            tensor("per_layer_token_embd.weight", e.ngram_embedding.weight)
            for k in ("key", "value"):
                tensor(p + "ple_" + k + ".weight", getattr(b, k + "_proj").weight)
            for k in ("key", "query", "conv"):
                tensor(p + "ple_norm_" + k + ".weight", 1 + getattr(b, "norm_" + k).weight)
            tensor(p + "ple_conv1d.weight", b.conv1d.weight[:, 0])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    prompt = [42, 43, 44, 2, 57, 8, 11, 32, 7, 13, 39, 81, 49, 0, 28, 34, 12, eos, 42]
    if mixed:
        # Both packed projections take q8 inputs. The router and shared expert still see x.
        model.model.layers[0].mlp.experts.register_forward_pre_hook(
            lambda _, args: (q8_inputs(args[0]), *args[1:])
        )
    with torch.inference_mode():
        logits = model(torch.tensor([prompt]), use_cache=False).logits[0]
    reference = reference_logits(model, prompt, (0,) if mixed else ())
    torch.testing.assert_close(reference, logits.double(), atol=2e-4, rtol=0)
    golden = {"prompt": prompt, "logits": logits.tolist(), "reference64": reference.tolist()}
    (out / (stem + ".json")).write_text(json.dumps(golden), encoding="utf-8")


if __name__ == "__main__":
    torch.set_num_threads(1)
    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)
    for parent in (False, True):
        write_fixture(target, parent)
    write_fixture(target, True, mixed=True)
