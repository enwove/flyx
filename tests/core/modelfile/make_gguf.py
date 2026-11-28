# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""GGUF test fixtures, written with the upstream `gguf` package (a test dependency only).

    python tests/core/modelfile/make_gguf.py ASSET_DIR      (after tests/e2e/make_tiny.py)

* the tiny models of make_tiny.py as GGUF files, laid out as llama.cpp's
  converter lays Qwen3.5 out (norm weights 1 + w, a head's gate rows after its
  q rows, value heads tiled over the key heads, the decay as -exp(A_log)): the
  golden outputs of the float64 reference must come out of them unchanged
* random blocks of every GGML block type with the values `gguf.quants`
  decodes them to (gguf_quant_cases.json), and a tiny Qwen3 per type whose
  matrices are such blocks (quant_<type>.gguf), and golden_test's prompts for
  them (golden_quant.json: each file held to its own plain decoding), and a
  LoRA adapter for them as Hugging Face PEFT saves one (peft_quant/)
* the tiny Qwen3.5 with its matrices quantized (tiny_q8.gguf: the block types
  of 32 weights, mixed, as its matrices are 64 and 128 wide)
* a tiny gpt-oss in MXFP4 and Q8_0 and the same in float32, with transformers' tokens for them (tiny_gpt_oss.py)
* a tiny Gemma 4 in Q8_0 and the same in float32, with transformers' tokens for them (tiny_gemma4.py)
* a small tokenizer of Gemma 4's kind (SentencePiece's BPE) with tokenizers' ids (gemma4_tokenizer.py)
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import gguf
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python" / "convert"))
from flyx_convert.fileformat import Reader, Writer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gemma4_tokenizer  # noqa: E402
import tiny_gemma4  # noqa: E402
import tiny_gpt_oss  # noqa: E402

QUANTS = ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K",
          "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ1_M", "IQ4_NL", "IQ4_XS", "NVFP4")
# Where a block keeps its binary16 scales (the others: one, at the start).
SCALES_AT = {"Q4_1": [0, 2], "Q5_1": [0, 2], "Q2_K": [80, 82], "Q3_K": [108], "Q4_K": [0, 2], "Q5_K": [0, 2],
             "Q6_K": [208]}


def set_scales(data, name, scales):
    """Finite scales for random blocks: data [blocks, bytes] uint8, scales [blocks, n] float16."""
    if name == "NVFP4":  # four unsigned e4m3 scales a block, not binary16: the subnormal ones, about |scales|
        mag = np.abs(np.ascontiguousarray(scales, np.float32)).reshape(len(data), -1)[:, :1]
        data[:, 0:4] = np.clip(np.rint(mag * 2 ** 10), 1, 7).astype(np.uint8)
        return
    bits = np.ascontiguousarray(scales, np.float16).view(np.uint16).reshape(len(data), -1)
    if name == "IQ1_M":  # no scale field: its four nibbles are the top nibbles of the four scale words
        for i in range(4):
            at = 49 + 2 * i
            data[:, at] = (data[:, at] & 0x0F) | (((bits[:, 0] >> (4 * i)) & 0xF) << 4).astype(np.uint8)
        return
    for j, at in enumerate(SCALES_AT.get(name, [0])):
        data[:, at:at + 2] = bits[:, j % bits.shape[1]].astype("<u2").view(np.uint8).reshape(-1, 2)


def nvfp4_blocks(a):
    """GGML NVFP4 blocks of a [rows, K] float32 matrix (K % 64 == 0), which gguf.quants decodes but does not write:
    per 16 values an unsigned e4m3 scale of about their largest / 6 (its value as GGML reads it, halved, times the
    table's doubled codes), each value the nearest e2m1 code; in a group the low nibbles hold values 0..7."""
    q = gguf.quants.NVFP4
    x = np.ascontiguousarray(a, np.float32).reshape(-1, 4, 16)
    d = q.fp32_to_ue4m3((np.abs(x).max(axis=2) / 6).astype(np.float32)).astype(np.uint8)
    step = q.ue4m3_to_fp32(d).astype(np.float32)[..., None] * 2  # what a code of magnitude 1 is worth
    v = np.where(step > 0, x / np.where(step > 0, step, 1), 0)
    grid = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)
    mag = np.abs(np.abs(v)[..., None] - grid).argmin(axis=-1)
    code = (mag | np.where((v < 0) & (mag > 0), 8, 0)).astype(np.uint8)
    qs = (code[..., :8] | (code[..., 8:] << 4)).astype(np.uint8)
    return np.concatenate([d, qs.reshape(-1, 32)], axis=1).reshape(a.shape[0], -1)


def export(source: Path, tokenizer_dir: Path, target: Path, *, storage="f32", pre="qwen35", nfc=True, quant=None,
           encode=None, part=None, shards=1):
    """The model of a .flyx file as GGUF.  quant: per tensor name, blocks to
    store instead of its values (they are in GGUF's layout already).  encode:
    per tensor name, the block type to quantize its values to (None: as storage says).
    part: "target" the model without its MTP layer, "mtp" a file of the MTP layer
    alone (with the embedding, head and final norm), as unsloth ships Qwen3.8-27B's.
    shards: when > 1, writes <target_stem>-00001-of-NNNNN.gguf .. NNNNN shards.
    """
    with Reader(str(source)) as r:
        _export_from_reader(r, tokenizer_dir, target, storage=storage, pre=pre, nfc=nfc, quant=quant,
                            encode=encode, part=part, shards=shards)


def _export_from_reader(r, tokenizer_dir, target, *, storage="f32", pre="qwen35", nfc=True, quant=None,
                        encode=None, part=None, shards=1):
    h, c = r.header, r.header["config"]
    arch = h["arch"]
    mtp = h.get("mtp") is not None and part != "target"
    first_path = target if shards == 1 else target.with_name(f"{target.stem}-00001-of-{shards:05d}.gguf")
    w = gguf.GGUFWriter(str(first_path), arch)
    if quant or encode:
        w.add_uint32("general.quantization_version", 2)
    total = c["layers"] + int(mtp)
    for key, value in {"block_count": total, "context_length": c["max_pos"], "embedding_length": c["hidden"],
                       "feed_forward_length": c["inter"], "attention.head_count": c["n_heads"],
                       "attention.head_count_kv": c["n_kv"], "attention.key_length": c["head_dim"],
                       "attention.value_length": c["head_dim"], "rope.dimension_count": c["rot_dim"]}.items():
        w.add_uint32(f"{arch}.{key}", value)
    w.add_float32(f"{arch}.rope.freq_base", c["rope_theta"])
    w.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", c["eps"])
    if arch == "qwen35":
        for key, value in {"nextn_predict_layers": int(mtp), "ssm.conv_kernel": c["conv_k"],
                           "ssm.state_size": c["lin_dk"], "ssm.group_count": c["lin_nk"],
                           "ssm.time_step_rank": c["lin_nv"], "ssm.inner_size": c["lin_nv"] * c["lin_dv"]}.items():
            w.add_uint32(f"{arch}.{key}", value)
        w.add_array(f"{arch}.attention.recurrent_layers", [t == "gdn" for t in c["layer_types"]] + ([False] if mtp else []))
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.add_string("tokenizer.ggml.pre", pre)
    w.add_bool("tokenizer.ggml.normalizer.nfc", nfc)
    tj = json.loads((tokenizer_dir / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = tj["model"]["vocab"]
    tokens = [None] * (max([*vocab.values(), *(a["id"] for a in tj["added_tokens"])]) + 1)
    types = [1] * len(tokens)
    for s, i in vocab.items():
        tokens[i] = s
    for a in tj["added_tokens"]:
        tokens[a["id"]] = a["content"]
        types[a["id"]] = 3 if a["special"] else 4
    w.add_array("tokenizer.ggml.tokens", tokens)
    w.add_array("tokenizer.ggml.token_type", types)
    w.add_array("tokenizer.ggml.merges", [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]])
    w.add_uint32("tokenizer.ggml.eos_token_id", vocab["<|im_end|>"])
    w.add_string("tokenizer.chat_template", r.blob("chat_template.jinja").decode())
    common = {"in_norm": "attn_norm.weight", "post_norm": "post_attention_norm.weight",
              "mlp.gate": "ffn_gate.weight", "mlp.up": "ffn_up.weight", "mlp.down": "ffn_down.weight",
              "attn.q": "attn_q.weight", "attn.k": "attn_k.weight", "attn.v": "attn_v.weight",
              "attn.o": "attn_output.weight", "attn.q_norm": "attn_q_norm.weight", "attn.k_norm": "attn_k_norm.weight",
              "gdn.qkv": "attn_qkv.weight", "gdn.z": "attn_gate.weight", "gdn.a": "ssm_alpha.weight",
              "gdn.b": "ssm_beta.weight", "gdn.out": "ssm_out.weight", "gdn.conv": "ssm_conv1d.weight",
              "gdn.A_log": "ssm_a", "gdn.dt_bias": "ssm_dt.bias", "gdn.norm": "ssm_norm.weight"}
    extra = {"mtp.fc": "nextn.eh_proj.weight", "mtp.norm_emb": "nextn.enorm.weight",
             "mtp.norm_hidden": "nextn.hnorm.weight", "mtp.norm": "nextn.shared_head_norm.weight"}
    built = []
    for name, rec in h["tensors"].items():
        in_mtp = name in extra or name.startswith("mtp.")
        if name.startswith("draft.") or (part == "target" and in_mtp) or \
                (part == "mtp" and not in_mtp and name not in {"target.embed", "target.lm_head", "target.norm"}):
            continue
        a = r.numpy(name)
        if rec["dtype"] == "bf16":
            a = (a.astype(np.uint32) << 16).view(np.float32)
        a = a.astype(np.float32).copy()
        norm = False
        if name in {"target.embed", "target.lm_head", "target.norm"}:
            dest = {"target.embed": "token_embd.weight", "target.lm_head": "output.weight", "target.norm": "output_norm.weight"}[name]
            norm = name == "target.norm"
        elif name in extra:
            dest = f"blk.{c['layers']}." + extra[name]
            norm = name != "mtp.fc"
        else:
            match = re.fullmatch(r"(target|mtp)\.L(\d+)\.(.*)", name)
            group, layer, kind = match.groups()
            dest = f"blk.{c['layers'] if group == 'mtp' else layer}." + common[kind]
            if arch == "qwen3" and kind == "post_norm":
                dest = f"blk.{layer}.ffn_norm.weight"
            norm = kind in {"in_norm", "post_norm", "attn.q_norm", "attn.k_norm"}
            if arch == "qwen35" and kind == "attn.q":
                a = a.reshape(2, c["n_heads"], c["head_dim"], -1).transpose(1, 0, 2, 3).reshape(a.shape)
            if kind.startswith("gdn.") and kind != "gdn.norm":
                nk, nv, dk, dv = (c[k] for k in ("lin_nk", "lin_nv", "lin_dk", "lin_dv"))
                # Construct the GGUF layout with array transposes, independently
                # of the C++ reader's per-channel addressing.
                def heads(x, dim):
                    return x.reshape(nk, nv // nk, dim, -1).transpose(1, 0, 2, 3).reshape(x.shape)
                if kind in {"gdn.qkv", "gdn.conv"}:
                    a[2 * nk * dk:] = heads(a[2 * nk * dk:], dv)
                elif kind == "gdn.out":
                    a = heads(a.T.copy(), dv).T.copy()
                elif kind == "gdn.z":
                    a = heads(a, dv)
                elif kind in {"gdn.a", "gdn.b", "gdn.dt_bias", "gdn.A_log"}:
                    if kind == "gdn.A_log":
                        a = -np.exp(a)
                    a = heads(a, 1)
        if arch == "qwen35" and norm:
            a = a + np.float32(1)
        if storage == "f16":
            a = a.astype(np.float16)
        if quant and name in quant:
            data, typ = quant[name]
            built.append((dest, data, typ))
        # Rows of whole blocks (not the conv kernels); encode is asked only then, as it hands out types in turn.
        elif (encode and a.ndim == 2 and a.shape[1] % 32 == 0
              and a.shape[1] % gguf.GGML_QUANT_SIZES[encode(name)][0] == 0):
            typ = encode(name)
            blocks = (nvfp4_blocks(a) if typ == gguf.GGMLQuantizationType.NVFP4
                      else gguf.quants.quantize(np.ascontiguousarray(a), typ))
            built.append((dest, blocks, typ))
        elif storage == "bf16" and a.ndim == 2:
            bits = a.view(np.uint32)
            bits = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
            built.append((dest, bits.view(np.uint8).reshape(a.shape[0], -1), gguf.GGMLQuantizationType.BF16))
        else:
            built.append((dest, np.ascontiguousarray(a), None))

    def finish_writer(writer, items):
        for dest, data, raw_dtype in items:
            if raw_dtype is None:
                writer.add_tensor(dest, data)
            else:
                writer.add_tensor(dest, data, raw_dtype=raw_dtype)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    if shards == 1:
        finish_writer(w, built)
        return
    w.add_uint16("split.no", 0)
    w.add_uint16("split.count", shards)
    w.add_int32("split.tensors.count", len(built))
    per = max(1, len(built) // shards)
    slices = [built[i * per:(i + 1) * per if i + 1 < shards else len(built)] for i in range(shards)]
    finish_writer(w, slices[0])
    for i in range(1, shards):
        spath = target.with_name(f"{target.stem}-{i + 1:05d}-of-{shards:05d}.gguf")
        sw = gguf.GGUFWriter(str(spath), arch)
        sw.add_uint16("split.no", i)
        sw.add_uint16("split.count", shards)
        sw.add_int32("split.tensors.count", len(built))
        finish_writer(sw, slices[i])


def quant_cases(out):
    rng = np.random.default_rng(20261110)
    cases = []
    for name in QUANTS:
        typ = getattr(gguf.GGMLQuantizationType, name)
        block, size = gguf.GGML_QUANT_SIZES[typ]
        blocks = 24
        data = rng.integers(0, 256, (blocks, size), dtype=np.uint8)
        # Every payload bit random; the scales finite, signed, and short enough
        # that the package's float32 arithmetic is exact (the values are then
        # the encoding's exact meaning, which the float64 reference computes).
        rows = np.arange(blocks, dtype=np.float32)[:, None]
        if name != "NVFP4":  # its e4m3 scales random too: every byte (0x7F, the sign bit) as GGML reads it
            set_scales(data, name, (rows - blocks / 2 + 0.5) * np.array([[1.0, 2.0]], np.float32) / 128)
        expected = gguf.quants.dequantize(data, typ).reshape(-1)
        cases.append({"type": name, "bytes": data.reshape(-1).tolist(), "expected": expected.tolist(), "block": block})
    (out / "gguf_quant_cases.json").write_text(json.dumps(cases), encoding="utf-8")


def quant_models(out):
    """A tiny Qwen3 per block type: every matrix random blocks of the type, and
    the same model with the weights `gguf.quants` decodes them to as float32.
    """
    with Reader(str(out / "tiny_qwen3.flyx")) as base:
        config = dict(base.header["config"], hidden=256, inter=256, layers=1, n_heads=8, n_kv=2,
                      head_dim=32, rot_dim=32, layer_types=["attn"])
        dims = {"target.embed": (config["vocab"], 256), "target.norm": (256,),
                "target.L0.in_norm": (256,), "target.L0.post_norm": (256,),
                "target.L0.attn.q_norm": (32,), "target.L0.attn.k_norm": (32,),
                "target.L0.attn.q": (256, 256), "target.L0.attn.k": (64, 256),
                "target.L0.attn.v": (64, 256), "target.L0.attn.o": (256, 256),
                "target.L0.mlp.gate": (256, 256), "target.L0.mlp.up": (256, 256),
                "target.L0.mlp.down": (256, 256)}
        rng = np.random.default_rng(83041)
        for kind in QUANTS:
            typ = getattr(gguf.GGMLQuantizationType, kind)
            block, size = gguf.GGML_QUANT_SIZES[typ]
            oracle = Writer()
            oracle.meta = {k: v for k, v in base.header.items() if k not in {"tensors", "blobs"}}
            oracle.meta["config"] = config
            quant = {}
            for name, shape in dims.items():
                if len(shape) == 1:
                    a = np.ones(shape, np.float32)
                else:
                    data = rng.integers(0, 256, (int(np.prod(shape)) // block, size), np.uint8)
                    set_scales(data, kind, rng.uniform(-0.001, 0.001, (len(data), 2)))
                    a = gguf.quants.dequantize(data, typ).reshape(shape)
                    quant[name] = (data.reshape(shape[0], -1), typ)
                oracle.add_tensor(name, "f32", shape, np.ascontiguousarray(a))
            for name in base.header["blobs"]:
                oracle.add_blob(name, base.blob(name))
            path = out / f"quant_{kind}.flyx"
            oracle.write(str(path))
            export(path, out / "hf_qwen3", out / f"quant_{kind}.gguf", quant=quant)
    # golden_test's prompts for these files (golden_test ASSET_DIR cuda quant quant_<type>.gguf): the reference has no
    # model of GGML blocks, so each file is held to its own plain decoding, its tree rounds of more than 16 rows
    # included (a round's block GEMMs and its head 16 rows at a time; tiny_qwen3_gguf.gguf is float32, read as bf16).
    golden = json.loads((out / "golden_qwen3.json").read_text(encoding="utf-8"))
    cases = [{"prompt": c["prompt"]} for c in golden["cases"]]
    (out / "golden_quant.json").write_text(json.dumps({"eos": golden["eos"], "plain_only": True, "cases": cases}),
                                           encoding="utf-8")
    peft_quant(out, dims)


def peft_quant(out, dims):
    """A LoRA adapter for the quant_<type>.gguf files as Hugging Face PEFT saves one (peft_quant/; the engine's --lora):
    rank 8, lora_alpha 16, on q, v, o, gate and down (none on k and up: zeros beside them), float16 values that bf16
    holds (the engines run adapters in bf16).  gguf_test.cpp holds the backends' block GEMMs beside it to the float64
    backend's."""
    from safetensors.numpy import save_file
    rng = np.random.default_rng(8)
    rank, alpha, tensors = 8, 16, {}

    def bf16(a):  # round to bf16 (nearest even), kept in float16
        b = np.ascontiguousarray(a, np.float32).view(np.uint32)
        b = ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
        return b.view(np.float32).astype(np.float16)

    for name, blk in (("q", "self_attn"), ("v", "self_attn"), ("o", "self_attn"), ("gate", "mlp"), ("down", "mlp")):
        n, k = dims[f"target.L0.{'attn' if blk == 'self_attn' else 'mlp'}.{name}"]
        p = f"base_model.model.model.layers.0.{blk}.{name}_proj"
        tensors[p + ".lora_A.weight"] = bf16(rng.standard_normal((rank, k)) / np.sqrt(k))
        tensors[p + ".lora_B.weight"] = bf16(rng.standard_normal((n, rank)) * 0.5 / np.sqrt(rank))
    d = out / "peft_quant"
    d.mkdir(exist_ok=True)
    save_file(tensors, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank, "lora_alpha": alpha, "use_rslora": False,
         "target_modules": ["q_proj", "v_proj", "o_proj", "gate_proj", "down_proj"], "bias": "none"}),
        encoding="utf-8")


def quantized_tiny(out):
    """The tiny Qwen3.5 (gated delta net, gated attention, MTP) with every matrix
    quantized, the types in turn: stacked projections end up of different types.
    The same tensors again as two files, the model without its MTP layer and
    the MTP layer alone (tiny_q8_target.gguf, tiny_q8_mtp.gguf).
    """
    types = [getattr(gguf.GGMLQuantizationType, n) for n in ("Q8_0", "Q5_1", "Q4_0", "Q5_0", "Q4_1")]
    seen = {}

    def encode(name):
        return seen.setdefault(name, types[len(seen) % len(types)])

    export(out / "tiny.flyx", out / "hf_target", out / "tiny_q8.gguf", encode=encode)
    export(out / "tiny.flyx", out / "hf_target", out / "tiny_q8_split.gguf", encode=encode, shards=2)
    export(out / "tiny.flyx", out / "hf_target", out / "tiny_q8_target.gguf", encode=encode, part="target")
    export(out / "tiny.flyx", out / "hf_target", out / "tiny_q8_mtp.gguf", encode=encode, part="mtp")
    # Every matrix in GGML's NVFP4, as akopytko/Qwen3.8-27B-NVFP4-GGUF has the 27B's.
    export(out / "tiny.flyx", out / "hf_target", out / "tiny_nvfp4.gguf",
           encode=lambda name: gguf.GGMLQuantizationType.NVFP4)


def explicit_head_dim(out):
    """Like the 27B family: attention width differs from hidden width, and
    hidden_size is not divisible by head_count. Keep an independent native
    file to check binding, upload and inference with those dimensions.
    """
    with Reader(str(out / "tiny.flyx")) as base:
        config = dict(base.header["config"], n_heads=3, n_kv=1)
        original_heads, hd = base.header["config"]["n_heads"], config["head_dim"]
        writer = Writer()
        writer.meta = {k: v for k, v in base.header.items() if k not in {"tensors", "blobs"}}
        writer.meta.update(config=config, draft=None)
        for name, rec in base.header["tensors"].items():
            if name.startswith("draft."):
                continue
            a = base.numpy(name).copy()
            if name.endswith("attn.q"):
                a = np.concatenate((a[:3 * hd], a[original_heads * hd:(original_heads + 3) * hd]))
            elif name.endswith(("attn.k", "attn.v")):
                a = a[:hd]
            elif name.endswith("attn.o"):
                a = a[:, :3 * hd]
            writer.add_tensor(name, rec["dtype"], a.shape, np.ascontiguousarray(a))
        for name in base.header["blobs"]:
            writer.add_blob(name, base.blob(name))
    path = out / "tiny_explicit_hd.flyx"
    writer.write(str(path))
    export(path, out / "hf_target", out / "tiny_explicit_hd.gguf")


def _bf16(writer, name, a):
    """A float32 matrix as GGUF's BF16, rounded to nearest even."""
    a = np.ascontiguousarray(a, np.float32)
    bits = a.view(np.uint32)
    bits = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    writer.add_tensor(name, bits.view(np.uint8).reshape(a.shape[0], -1), raw_dtype=gguf.GGMLQuantizationType.BF16)


def _dflash2_keys(w, d, *, causal=False, taps=None, pattern=None, extra=()):
    """The metadata of a llama.cpp arch='dflash' GGUF, from a canonical "draft" config."""
    w.add_uint32("dflash.block_count", d["layers"])
    w.add_uint32("dflash.embedding_length", d["hidden"])
    w.add_uint32("dflash.feed_forward_length", d["inter"])
    w.add_uint32("dflash.attention.head_count", d["n_heads"])
    w.add_uint32("dflash.attention.head_count_kv", d["n_kv"])
    w.add_uint32("dflash.attention.key_length", d["head_dim"])
    w.add_uint32("dflash.attention.value_length", d["head_dim"])
    w.add_bool("dflash.attention.causal", causal)
    w.add_float32("dflash.rope.freq_base", d["rope_theta"])
    w.add_float32("dflash.attention.layer_norm_rms_epsilon", d["eps"])
    w.add_uint32("dflash.block_size", d["block"])
    w.add_uint32("dflash.conv_kernel_size", d["conv_taps"])
    w.add_uint32("dflash.conv_group_size", d["conv_group"])
    w.add_uint32("dflash.selector_rank", d["sel_rank"])
    w.add_uint32("dflash.selector_top_k", d["sel_topk"])
    w.add_array("dflash.target_layers", [int(t) for t in (taps or d["taps"])])
    win = int(d.get("window", 0) or 0)
    if win > 0:
        w.add_uint32("dflash.attention.sliding_window", win)
        if pattern:
            w.add_array("dflash.attention.sliding_window_pattern", [bool(p) for p in pattern])
    for key, value in extra:
        (w.add_bool if isinstance(value, bool) else w.add_float32)(key, value)
    w.add_uint32("tokenizer.ggml.mask_token_id", d["mask_token"])


def export_dflash2_gguf(source: Path, target: Path, *, encode=None, **keys):
    """Export a canonical DFlash2 drafter .flyx file as a llama.cpp arch='dflash' GGUF.  encode: a GGML block type
    for a matrix whose rows are whole blocks of it (others stay BF16).  keys: what the metadata says other than the
    drafter's own config (_dflash2_keys: for the files the engine refuses)."""
    with Reader(str(source)) as r:
        d = r.header["draft"]
        w = gguf.GGUFWriter(str(target), "dflash")
        if encode:
            w.add_uint32("general.quantization_version", 2)
        _dflash2_keys(w, d, **keys)

        top_map = {
            "draft.fc": "fc.weight",
            "draft.hidden_norm": "enc.output_norm.weight",
            "draft.norm": "output_norm.weight",
            "draft.sel.pred": "selector_predecessor.weight",
            "draft.sel.succ": "selector_successor.weight",
            "draft.sel.hid": "selector_hidden.weight",
        }
        layer_map = {
            "in_norm": "attn_norm.weight",
            "post_norm": "ffn_norm.weight",
            "attn.q_norm": "attn_q_norm.weight",
            "attn.k_norm": "attn_k_norm.weight",
            "attn.q": "attn_q.weight",
            "attn.k": "attn_k.weight",
            "attn.v": "attn_v.weight",
            "attn.o": "attn_output.weight",
            "mlp.gate": "ffn_gate.weight",
            "mlp.up": "ffn_up.weight",
            "mlp.down": "ffn_down.weight",
            "attn_conv.proj": "attn_conv_proj.weight",
            "attn_conv.base": "attn_conv_base",
            "mlp_conv.proj": "ffn_conv_proj.weight",
            "mlp_conv.base": "ffn_conv_base",
        }
        for name, rec in r.header["tensors"].items():
            if name in top_map:
                dest = top_map[name]
            else:
                m = re.fullmatch(r"draft\.L(\d+)\.(.*)", name)
                if not m:
                    continue
                dest = f"blk.{m.group(1)}." + layer_map[m.group(2)]
            a = r.numpy(name)
            if rec["dtype"] == "bf16":
                a = (a.astype(np.uint32) << 16).view(np.float32)
            a = np.ascontiguousarray(a.astype(np.float32))
            if (encode and a.ndim == 2 and a.shape[1] % 32 == 0
                    and a.shape[1] % gguf.GGML_QUANT_SIZES[encode(name)][0] == 0):
                typ = encode(name)
                blocks = gguf.quants.quantize(a, typ)
                w.add_tensor(dest, blocks, raw_dtype=typ)
            elif a.ndim == 2:
                _bf16(w, dest, a)
            else:
                w.add_tensor(dest, a)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()


def random_dflash2_gguf(model: Path, target: Path, seed: int):
    """A DFlash2 GGUF drafter of random weights for a dense model (.flyx: its width, vocabulary and layer count),
    reading its first layer's output and the one before its last's."""
    with Reader(str(model)) as r:
        c = r.header["config"]
    H, V, target_layers = int(c["hidden"]), int(c["vocab"]), int(c["layers"])
    assert target_layers >= 3, "the drafter taps two layers before the model's last"
    layers, nh, nkv, hd, inter = 2, 4, 2, 32, max(128, H * 2)
    ktaps, group, rank = 2, 16, 16
    taps = [1, target_layers - 1]
    G = H // group
    rng = np.random.default_rng(seed)

    def norm(n):
        return np.ascontiguousarray(1.0 + 0.1 * rng.normal(0, 1, (n,)), np.float32)

    w = gguf.GGUFWriter(str(target), "dflash")
    _dflash2_keys(w, {"layers": layers, "hidden": H, "inter": inter, "n_heads": nh, "n_kv": nkv, "head_dim": hd,
                      "rope_theta": 10000.0, "eps": 1e-6, "block": 8, "conv_taps": ktaps, "conv_group": group,
                      "sel_rank": rank, "sel_topk": 16, "taps": taps, "mask_token": 3})
    _bf16(w, "fc.weight", rng.normal(0, (len(taps) * H) ** -0.5, (H, len(taps) * H)))
    w.add_tensor("enc.output_norm.weight", norm(H))
    w.add_tensor("output_norm.weight", norm(H))
    _bf16(w, "selector_predecessor.weight", rng.normal(0, 0.5, (V, rank)))
    _bf16(w, "selector_successor.weight", rng.normal(0, 0.5, (V, rank)))
    _bf16(w, "selector_hidden.weight", rng.normal(0, H ** -0.5, (rank, H)))
    for layer in range(layers):
        p = f"blk.{layer}."
        w.add_tensor(p + "attn_norm.weight", norm(H))
        w.add_tensor(p + "ffn_norm.weight", norm(H))
        w.add_tensor(p + "attn_q_norm.weight", norm(hd))
        w.add_tensor(p + "attn_k_norm.weight", norm(hd))
        _bf16(w, p + "attn_q.weight", rng.normal(0, H ** -0.5, (nh * hd, H)))
        _bf16(w, p + "attn_k.weight", rng.normal(0, H ** -0.5, (nkv * hd, H)))
        _bf16(w, p + "attn_v.weight", rng.normal(0, H ** -0.5, (nkv * hd, H)))
        _bf16(w, p + "attn_output.weight", rng.normal(0, (nh * hd) ** -0.5, (H, nh * hd)))
        _bf16(w, p + "ffn_gate.weight", rng.normal(0, H ** -0.5, (inter, H)))
        _bf16(w, p + "ffn_up.weight", rng.normal(0, H ** -0.5, (inter, H)))
        _bf16(w, p + "ffn_down.weight", rng.normal(0, inter ** -0.5, (H, inter)))
        for kind in ("attn", "ffn"):
            base = (0.2 * rng.normal(0, 1, (2, ktaps, H))).astype(np.float32)
            base[:, 0] += 1.0
            w.add_tensor(p + f"{kind}_conv_base", np.ascontiguousarray(base))
            _bf16(w, p + f"{kind}_conv_proj.weight", rng.normal(0, 0.3 * H ** -0.5, (2 * ktaps * G, H)))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def dflash2_ggufs(out: Path):
    """DFlash2 drafters as llama.cpp's GGUF files (arch "dflash"), beside the tiny models:
    tiny_draft*.gguf: tiny.flyx's drafter in BF16, with a sliding window, in Q8_0, and with the types mixed as a
      Q4_K_M file mixes them (the stacked q|k|v and context k|v of more than one type);
    tiny_qwen3_draft.gguf, tiny_llama_draft.gguf: random drafters for the tiny Qwen3 and the tiny Llama;
    tiny_draft_softcap.gguf, tiny_draft_last_layer.gguf: tiny.flyx's drafter with a soft cap on its selector's
      logits, and reading the model's final hidden state;
    tiny_draft_refused_*.gguf: tiny.flyx's drafter saying what the engine does not run (a DSpark drafter's Markov
      context: not beside a dense model)."""
    types = gguf.GGMLQuantizationType
    export_dflash2_gguf(out / "tiny_draft.flyx", out / "tiny_draft.gguf")
    with Reader(str(out / "tiny_draft_window.flyx")) as r:
        sliding = r.header["draft"]["layers"]
    assert sliding >= 2, "a pattern that is not uniform needs two layers"
    export_dflash2_gguf(out / "tiny_draft_window.flyx", out / "tiny_draft_window.gguf", pattern=[True] * sliding)
    export_dflash2_gguf(out / "tiny_draft.flyx", out / "tiny_draft_q8.gguf", encode=lambda _: types.Q8_0)
    export_dflash2_gguf(out / "tiny_draft.flyx", out / "tiny_draft_mixed.gguf",
                        encode=lambda name: types.Q8_0 if name.endswith(("attn.v", "mlp.down")) else types.Q4_0)
    random_dflash2_gguf(out / "tiny_qwen3.flyx", out / "tiny_qwen3_draft.gguf", 20261124)
    random_dflash2_gguf(out / "tiny_llama.flyx", out / "tiny_llama_draft.gguf", 20261125)

    with Reader(str(out / "tiny_draft.flyx")) as r:
        d = r.header["draft"]
    refused = {
        "causal": {"causal": True},
        "dspark": {"extra": [("dflash.has_markov_context", True)]},
        "anchor": {"extra": [("dflash.sample_from_anchor", True)]},
        "confidence": {"extra": [("dflash.has_confidence_head", True)]},
    }
    export_dflash2_gguf(out / "tiny_draft.flyx", out / "tiny_draft_softcap.gguf",
                        extra=[("dflash.final_logit_softcapping", 30.0)])
    export_dflash2_gguf(out / "tiny_draft.flyx", out / "tiny_draft_last_layer.gguf",
                        taps=d["taps"][:-1] + [d["target"]["layers"]])
    for name, keys in refused.items():
        export_dflash2_gguf(out / "tiny_draft.flyx", out / f"tiny_draft_refused_{name}.gguf", **keys)
    export_dflash2_gguf(out / "tiny_draft_window.flyx", out / "tiny_draft_refused_pattern.gguf",
                        pattern=[i % 2 == 0 for i in range(sliding)])


def main(out):
    out = Path(out)
    export(out / "tiny.flyx", out / "hf_target", out / "tiny_gguf.gguf")
    export(out / "tiny.flyx", out / "hf_target", out / "tiny_split.gguf", shards=3)
    export(out / "tiny.flyx", out / "hf_target", out / "tiny_bf16.gguf", storage="bf16")
    export(out / "tiny_qwen3.flyx", out / "hf_qwen3", out / "tiny_qwen3_gguf.gguf", pre="qwen35")
    export(out / "tiny_qwen3.flyx", out / "hf_qwen3", out / "tiny_qwen3_f16.gguf", storage="f16", pre="qwen35")
    # A real qwen2 split, with and without NFC, checked against tokenizers.
    from tokenizers import Regex, Tokenizer, normalizers, pre_tokenizers
    from flyx_convert.tokenizer_export import QWEN2_SPLIT
    for nfc in (False, True):
        name = "qwen2_nfc" if nfc else "qwen2_none"
        tok = Tokenizer.from_file(str(out / "hf_qwen3" / "tokenizer.json"))
        tok.normalizer = normalizers.NFC() if nfc else None
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(QWEN2_SPLIT), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        texts = ["a\u0301\u0308b", "\u0301\u0308hello", "हिन्दी తెలుగు", "12345678", "  \t\r\nnext", "!!!\u0301a",
                 "I'M we'll THEY'RE", "<think>x<|im_end|>", "", "e\u0301\u0000e\u0301x"]
        cases = [{"text": s, "ids": tok.encode(s).ids} for s in texts]
        (out / f"{name}.json").write_text(json.dumps(cases), encoding="utf-8")
        export(out / "tiny_qwen3.flyx", out / "hf_qwen3", out / f"{name}.gguf", pre="qwen2", nfc=nfc)
    quant_cases(out)
    quant_models(out)
    quantized_tiny(out)
    explicit_head_dim(out)
    dflash2_ggufs(out)
    tiny_gpt_oss.write(out)
    tiny_gpt_oss.write(out, quant=False, stem="gpt_oss_f32")
    tiny_gemma4.write(out)
    tiny_gemma4.write(out, quant=False, stem="gemma4_f32")
    tiny_gemma4.write(out, stem="gemma4_moe", moe=True)
    tiny_gemma4.write(out, quant=False, stem="gemma4_moe_f32", moe=True)
    gemma4_tokenizer.write(out)
    print("wrote GGUF model, tokenizer and independent quantization fixtures")


if __name__ == "__main__":
    main(sys.argv[1])
