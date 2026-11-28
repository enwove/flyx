# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""A small tokenizer of Gemma 4's kind (SentencePiece's BPE as Hugging Face's tokenizers runs it: every space written
U+2581, BPE over a segment's characters by merge rank, <0xNN> byte tokens for a character the vocabulary lacks), as
llama.cpp's converter writes Gemma 4's into a GGUF file (tokenizer.ggml.model "gemma4"), and tokenizers' own ids.

    gemma4_tok.gguf   the vocabulary (special tokens, the 256 byte tokens, a few hundred pieces trained on the texts
                      below, none mixing a newline with another character, as Gemma 4's own) and its merges;
                      add_bos_token, as Gemma 4's files have it (a raw prompt still takes tokenizers' tokens, no
                      <bos> first: the chat template writes it)
    gemma4_tok.json   tokenizer_test's cases: each text's ids from tokenizers, alone and as a raw prompt
"""
from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path

import gguf
import numpy as np

SPECIALS = ["<pad>", "<eos>", "<bos>", "<unk>", "<|turn>", "<turn|>"]
VISIBLE = ["<|tool_call>", "<tool_call|>", "<|channel>", "<channel|>"]  # read as text (llama.cpp's USER_DEFINED)
CORPUS = [
    "the quick brown fox jumps over the lazy dog", "The rain in Spain stays mainly in the plain.",
    "def tokenize(text):\n    return [piece for piece in text.split()]\n", "numbers 0123456789 and 3.14159",
    "a sentence, with punctuation; and more!  Double  spaces too.", "naïve façade café über", "日本語 中文 한국어",
    "line one\nline two\n\nline four\n\n\n", "<html><body>hello</body></html>", "tokens tokens tokens tokenizer",
]
TEXTS = [
    "the quick brown fox", " leading space", "trailing space ", "  two  spaces  ", "\n", "\n\n\n\n", "a\nb\n\nc",
    "line one\r\nline two\r\n", "tab\tseparated", "naïve façade", "é decomposed, é composed",
    "日本語のテキスト", "emoji 😀 and \U00010348", "<|turn>user\nhi<turn|>", "<bos> written",
    "<|tool_call>call:f{}<tool_call|> <|channel>thought<channel|>", "▁ a metaspace", "\x00\x01 bytes",
    "the " * 70, "x" * 300,
]


def inputs_digest() -> str:
    """What the files are made from: this script and the packages that make them."""
    h = hashlib.sha256(Path(__file__).read_bytes())
    for pkg in ("tokenizers", "gguf", "numpy"):
        h.update(f"{pkg} {metadata.version(pkg)}\0".encode())
    return h.hexdigest()


def write(out: Path):
    stamp = out / "gemma4_tok.stamp"
    if stamp.exists() and stamp.read_text() == inputs_digest() and (out / "gemma4_tok.gguf").exists() and \
            (out / "gemma4_tok.json").exists():
        return
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    # Trained with newlines apart (Gemma 4's vocabulary mixes none with another character), then run as Gemma 4's
    # tokenizer.json runs: the spaces replaced, no split (its Split on " " finds none left), byte fallback.
    trainer_tok = Tokenizer(models.BPE())
    trainer_tok.normalizer = normalizers.Replace(" ", "▁")
    trainer_tok.pre_tokenizer = pre_tokenizers.Split(Regex("\n+"), "isolated")
    trainer_tok.train_from_iterator(CORPUS * 4, trainers.BpeTrainer(vocab_size=420, show_progress=False))
    learned = json.loads(trainer_tok.to_str())["model"]
    vocab = {t: i for i, t in enumerate(SPECIALS + VISIBLE)}
    for b in range(256):
        vocab[f"<0x{b:02X}>"] = len(vocab)
    for piece, _ in sorted(learned["vocab"].items(), key=lambda kv: kv[1]):
        if piece not in vocab:
            vocab[piece] = len(vocab)
    merges = [tuple(m) if isinstance(m, list) else tuple(m.split(" ")) for m in learned["merges"]]
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, byte_fallback=True, fuse_unk=True, unk_token="<unk>"))
    tok.normalizer = normalizers.Replace(" ", "▁")
    tok.pre_tokenizer = pre_tokenizers.Split(" ", "merged_with_previous")
    tok.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()])
    tok.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in SPECIALS])
    tok.add_tokens([AddedToken(t, special=False, normalized=False) for t in VISIBLE])

    tokens = [t for t, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    types = [3] * len(SPECIALS) + [4] * len(VISIBLE) + [6] * 256 + [1] * (len(tokens) - len(SPECIALS) - len(VISIBLE)
                                                                         - 256)
    w = gguf.GGUFWriter(str(out / "gemma4_tok.gguf"), "gemma4")
    w.add_name("gemma4-tokenizer")
    w.add_string("tokenizer.ggml.model", "gemma4")
    w.add_array("tokenizer.ggml.tokens", tokens)
    w.add_array("tokenizer.ggml.scores", [-1000.0] * len(tokens))
    w.add_array("tokenizer.ggml.token_type", types)
    w.add_array("tokenizer.ggml.merges", [f"{a} {b}" for a, b in merges])
    w.add_uint32("tokenizer.ggml.bos_token_id", vocab["<bos>"])
    w.add_uint32("tokenizer.ggml.eos_token_id", vocab["<turn|>"])
    w.add_uint32("tokenizer.ggml.unknown_token_id", vocab["<unk>"])
    w.add_uint32("tokenizer.ggml.padding_token_id", vocab["<pad>"])
    w.add_bool("tokenizer.ggml.add_bos_token", True)
    w.add_bool("tokenizer.ggml.add_space_prefix", False)
    w.add_tensor("token_embd.weight", np.zeros((len(tokens), 4), np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    cases = []
    for text in TEXTS:
        ids = tok.encode(text, add_special_tokens=False).ids
        cases.append({"text": text, "ids": ids, "prompt_ids": tok.encode(text).ids})
    (out / "gemma4_tok.json").write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    stamp.write_text(inputs_digest())


if __name__ == "__main__":
    import sys
    write(Path(sys.argv[1]))
