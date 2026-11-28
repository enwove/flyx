# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Turn a HuggingFace tokenizer.json into the compact tables the engine loads.

The engine never parses tokenizer.json.  It gets:
  tokenizer.vocab   u32 offsets[n+1] followed by the raw bytes of every token
  tokenizer.merges  u32 triples (left id, right id, merged id) in rank order
  tokenizer.added   JSON list of added tokens {id, content, special}
and a small description in the header: the pre-tokenizer, the normalizer, and
the tokens the post-processor puts before a text (prefix_ids, MiniCPM5's
`<s>`; absent: none).  Only the pipelines we implement in C++ are accepted;
anything else fails here, at conversion time.  Two kinds: byte-level BPE (GPT-2's
alphabet), and SentencePiece's BPE (Gemma 4: every space written U+2581, byte
fallback tokens <0x00>..<0xFF>, whose ids go in the header as byte_ids).
"""
from __future__ import annotations

import json
import struct

# The pre-tokenizer split used by Qwen3.5 (letters may carry combining marks).
QWEN35_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"
                r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
QWEN2_SPLIT = QWEN35_SPLIT.replace(r"[\p{L}\p{M}]+", r"\p{L}+").replace(r"\p{M}", "")
# MiniCPM5: numbers cut into groups of up to three from the left first, then Qwen2's split with a run of numbers as
# one, on each piece of the first.
MINICPM5_NUMBERS = r"\p{N}{1,3}"
MINICPM5_SPLIT = QWEN2_SPLIT.replace(r"|\p{N}|", r"|\p{N}+|")


def byte_level_alphabet() -> dict[int, str]:
    """Byte -> printable character map used by byte-level BPE (GPT-2 scheme)."""
    keep = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    table, extra = {}, 0
    for b in range(256):
        if b in keep:
            table[b] = chr(b)
        else:
            table[b] = chr(256 + extra)
            extra += 1
    return table


# The regex splits before the byte-level step -> the pre-tokenizer the engine implements (core/tokenizer).
PRETOKENIZERS = {(QWEN35_SPLIT,): "qwen35", (QWEN2_SPLIT,): "qwen2", (MINICPM5_NUMBERS, MINICPM5_SPLIT): "minicpm5"}


def _split_regex(step: dict) -> str:
    if step.get("type") != "Split" or step.get("behavior") != "Isolated" or step.get("invert"):
        raise ValueError(f"unsupported split: {step}")
    return (step.get("pattern") or {}).get("Regex")


METASPACE = "\u2581"
# The marks of Gemma 4's replies that its GGUF files keep as text (llama.cpp's USER_DEFINED): not special here either,
# so the two formats read a reply's marks alike and a reply grammar matches them as text.
GEMMA4_TEXT_MARKS = {"<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", "<|tool_response>",
                     "<tool_response|>", "<|\"|>"}


def _is_metaspace(tj: dict) -> bool:
    """Gemma 4's pipeline: spaces replaced by U+2581, a split on spaces (none left) or none, BPE with byte fallback."""
    norm, pre, model = tj.get("normalizer") or {}, tj.get("pre_tokenizer"), tj.get("model") or {}
    if norm.get("type") != "Replace" or (norm.get("pattern") or {}).get("String") != " " or \
            norm.get("content") != METASPACE:
        return False
    if pre is not None and not (pre.get("type") == "Split" and (pre.get("pattern") or {}).get("String") == " " and
                                not pre.get("invert")):
        return False
    return model.get("type") == "BPE" and bool(model.get("byte_fallback"))


def _check_metaspace(tj: dict) -> dict:
    model = tj["model"]
    if model.get("dropout") or model.get("continuing_subword_prefix") or model.get("end_of_word_suffix") or \
            model.get("ignore_merges"):
        raise ValueError(f"unsupported SentencePiece BPE: { {k: model.get(k) for k in ('dropout', 'ignore_merges')} }")
    vocab = model["vocab"]
    missing = [b for b in range(256) if f"<0x{b:02X}>" not in vocab]
    if missing:
        raise ValueError(f"byte fallback without the token of byte {missing[0]}")
    if any(" " in t for t in vocab):
        raise ValueError("a SentencePiece token with a space in it (spaces are U+2581)")
    info = {"pretokenizer": "gemma4", "normalizer": "none", "ignore_merges": False,
            "byte_ids": [int(vocab[f"<0x{b:02X}>"]) for b in range(256)]}
    prefix = _prefix_ids(tj.get("post_processor"))
    if prefix:
        info["prefix_ids"] = prefix
    return info


def _check_pipeline(tj: dict) -> dict:
    if _is_metaspace(tj):
        return _check_metaspace(tj)
    norm = tj.get("normalizer")
    if norm is not None and norm.get("type") != "NFC":
        raise ValueError(f"unsupported normalizer: {norm}")
    pre = tj.get("pre_tokenizer") or {}
    seq = pre.get("pretokenizers") if pre.get("type") == "Sequence" else None
    if not seq or len(seq) < 2:
        raise ValueError(f"unsupported pre_tokenizer: {pre}")
    splits, bl = tuple(_split_regex(step) for step in seq[:-1]), seq[-1]
    if splits not in PRETOKENIZERS:
        raise ValueError(f"unsupported split regexes: {splits!r}")
    if bl.get("type") != "ByteLevel" or bl.get("add_prefix_space") or bl.get("use_regex"):
        raise ValueError(f"unsupported byte-level step: {bl}")
    model = tj["model"]
    if model.get("type") != "BPE" or model.get("byte_fallback") or model.get("dropout"):
        raise ValueError(f"unsupported model: { {k: model.get(k) for k in ('type', 'byte_fallback', 'dropout')} }")
    if model.get("continuing_subword_prefix") or model.get("end_of_word_suffix"):
        raise ValueError("subword prefixes/suffixes are not supported")
    info = {"pretokenizer": PRETOKENIZERS[splits],
            "normalizer": "nfc" if norm else "none",
            "ignore_merges": bool(model.get("ignore_merges", False))}
    prefix = _prefix_ids(tj.get("post_processor"))
    if prefix:
        info["prefix_ids"] = prefix
    return info


def _prefix_ids(post: dict | None) -> list[int]:
    """The tokens the post-processor puts before a text (MiniCPM5's TemplateProcessing: `<s>` $A): the engine puts
    them before a raw prompt.  A ByteLevel post-processor only moves offsets.  Only the single-text form counts: the
    engine never encodes a pair.  Tokens after the text, or any other post-processor, are refused."""
    if post is None:
        return []
    steps = post.get("processors", []) if post.get("type") == "Sequence" else [post]
    templates = [s for s in steps if s.get("type") != "ByteLevel"]
    if len(templates) > 1 or any(s.get("type") != "TemplateProcessing" for s in templates):
        raise ValueError(f"unsupported post_processor: {post}")
    prefix: list[int] = []
    for step in templates:
        single, specials = step.get("single") or [], step.get("special_tokens") or {}
        if not single or (single[-1].get("Sequence") or {}).get("id") != "A":
            raise ValueError(f"unsupported post_processor (only tokens before the text): {single}")
        for piece in single[:-1]:
            name = (piece.get("SpecialToken") or {}).get("id")
            if name not in specials:
                raise ValueError(f"unsupported post_processor piece: {piece}")
            prefix += [int(i) for i in specials[name]["ids"]]
    return prefix


def export(tokenizer_json_path: str):
    """Returns (header_info, {blob_name: bytes})."""
    with open(tokenizer_json_path, encoding="utf-8") as f:
        tj = json.load(f)
    info = _check_pipeline(tj)
    metaspace = info["pretokenizer"] == "gemma4"
    inv = {c: b for b, c in byte_level_alphabet().items()}
    vocab: dict[str, int] = tj["model"]["vocab"]
    added = tj.get("added_tokens", [])
    n = max([*vocab.values(), *(a["id"] for a in added)]) + 1
    toks: list[bytes] = [b""] * n
    byte_of = {i: b for b, i in enumerate(info.get("byte_ids", []))}
    for s, i in vocab.items():
        if metaspace:  # a byte token's byte, else the text, spaces back
            toks[i] = bytes([byte_of[i]]) if i in byte_of else s.replace(METASPACE, " ").encode("utf-8")
        else:
            toks[i] = bytes(inv[ch] for ch in s)
    for a in added:
        toks[a["id"]] = a["content"].encode("utf-8")
    offs, acc = [0], 0
    for t in toks:
        acc += len(t)
        offs.append(acc)
    vocab_blob = struct.pack(f"<{n + 1}I", *offs) + b"".join(toks)

    triples = []
    for m in tj["model"]["merges"]:
        a, b = (m.split(" ", 1) if isinstance(m, str) else m)
        triples += [vocab[a], vocab[b], vocab[a + b]]
    merges_blob = struct.pack(f"<{len(triples)}I", *triples)
    def special(a):
        return bool(a.get("special")) and not (metaspace and a["content"] in GEMMA4_TEXT_MARKS)
    added_blob = json.dumps([{"id": a["id"], "content": a["content"], "special": special(a)}
                             for a in added], ensure_ascii=False).encode()
    # The engine writes the prefix as its tokens' text where special tokens are read: each must be one.
    special = {a["id"] for a in added if special(a)}
    if any(i not in special for i in info.get("prefix_ids", [])):
        raise ValueError(f"the post-processor's tokens {info['prefix_ids']} are not all special added tokens")
    info.update(n_tokens=n, n_merges=len(triples) // 3)
    return info, {"tokenizer.vocab": vocab_blob, "tokenizer.merges": merges_blob, "tokenizer.added": added_blob}
