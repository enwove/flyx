# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""How much of a coding agent's output drafts looked up in its own context would cover.

    python bench/replay/lookup_coverage.py --tokenizer ~/models/Qwen3.5-4B/tokenizer.json SESSION.jsonl|DIR ...

Reads Claude Code transcripts (JSONL, one event per line; message.content blocks of
type text, tool_use, tool_result). In each session the agent's output is the text and
the tool calls it wrote; its context is everything before (user messages, tool
results, earlier output). Drafting by lookup (prompt lookup decoding, SuffixDecoding)
is simulated: at each step the longest n-gram (n = 8 down to --min-n) that ends at
the current position and occurred before proposes the tokens that followed its
latest occurrence, up to --max-draft; the round accepts the proposed tokens that
match the output, plus the target's own token. Without a match, the model's drafter
produces the token.

Reported per policy (use lookup when the matched n-gram has at least L tokens):
the share of output tokens produced in lookup rounds, accepted drafts per lookup
round, and the decode time against the model drafter alone, with a round-cost model
(--mtp-tok-s: the model drafter's speed; a lookup round with k drafts costs
--base-ms + k * --row-ms). Only aggregate counts are printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content if isinstance(content, list) else []


def block_text(b) -> str:
    t = b.get("type")
    if t == "text":
        return b.get("text", "")
    if t == "tool_result":
        return "".join(x.get("text", "") for x in blocks(b.get("content")) if isinstance(x, dict))
    return ""


def session_pieces(path: Path):
    """(is_output, text) in order: what the agent wrote, and what it read."""
    for line in path.open(encoding="utf-8", errors="replace"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        m = d.get("message")
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        for b in blocks(m.get("content")):
            if not isinstance(b, dict):
                continue
            if role == "assistant":
                if b.get("type") == "text" and b.get("text"):
                    yield True, b["text"]
                elif b.get("type") == "tool_use":
                    yield True, json.dumps({"name": b.get("name"), "arguments": b.get("input")}, ensure_ascii=False)
            elif role == "user":
                t = block_text(b)
                if t:
                    yield False, t


def simulate(stream, is_out, policies, min_n, max_draft):
    """Per policy L: [lookup_rounds, lookup_tokens, accepted, model_tokens]."""
    stats = {L: [0, 0, 0, 0] for L in policies}
    index = {}  # n-gram -> end position of its latest occurrence
    n_max = 8

    def add(i):  # index the n-grams that end right before position i
        for n in range(min_n, n_max + 1):
            if i >= n:
                index[tuple(stream[i - n:i])] = i

    # Each policy advances through the output on its own: its rounds differ.
    cursor = {L: 0 for L in policies}  # next output position not yet produced, per policy
    for i in range(len(stream)):
        if is_out[i]:
            for L in policies:
                if cursor[L] > i:
                    continue  # produced already by an earlier lookup round of this policy
                best_n, p = 0, -1
                for n in range(n_max, min_n - 1, -1):
                    if i >= n:
                        q = index.get(tuple(stream[i - n:i]))
                        if q is not None and q < i:
                            best_n, p = n, q
                            break
                s = stats[L]
                if best_n >= L:
                    k = min(max_draft, i - p)  # the continuation must be text seen already
                    acc = 0
                    while acc < k and i + acc < len(stream) and is_out[i + acc] and stream[p + acc] == stream[i + acc]:
                        acc += 1
                    s[0] += 1
                    s[1] += acc + 1
                    s[2] += acc
                    cursor[L] = i + acc + 1
                else:
                    s[3] += 1
                    cursor[L] = i + 1
        add(i)  # after the lookup: the n-grams ending here now have this as their latest occurrence
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json of the target model")
    ap.add_argument("--min-n", type=int, default=3)
    ap.add_argument("--max-draft", type=int, default=15)
    ap.add_argument("--max-tokens", type=int, default=600000, help="per session (the start of long ones)")
    ap.add_argument("--mtp-tok-s", type=float, default=459.0, help="the model drafter's decode speed")
    ap.add_argument("--base-ms", type=float, default=4.7, help="a round with no drafts")
    ap.add_argument("--row-ms", type=float, default=0.1, help="each verified draft row")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.expanduser(a.tokenizer))
    files = []
    for p in a.paths:
        p = Path(os.path.expanduser(p))
        files += sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
    policies = [3, 4, 6, 8]
    total = {L: [0, 0, 0, 0] for L in policies}
    out_tokens = sessions = 0
    for f in files:
        stream, is_out = [], []
        for out, text in session_pieces(f):
            ids = tok.encode(text, add_special_tokens=False).ids
            stream += ids
            is_out += [out] * len(ids)
            if len(stream) >= a.max_tokens:
                break
        if not any(is_out):
            continue
        sessions += 1
        out_tokens += sum(is_out)
        for L, s in simulate(stream, is_out, policies, a.min_n, a.max_draft).items():
            for k in range(4):
                total[L][k] += s[k]
    print(f"{sessions} sessions, {out_tokens:,} output tokens (of agent text and tool calls), max draft {a.max_draft}")
    model_ms = out_tokens * 1000.0 / a.mtp_tok_s
    print(f"model drafter alone: {model_ms / 1000:.1f} s at {a.mtp_tok_s:.0f} tok/s")
    for L in policies:
        rounds, lookup_tok, acc, model_tok = total[L]
        t = rounds * (a.base_ms + a.max_draft * a.row_ms) + model_tok * 1000.0 / a.mtp_tok_s
        print(f"lookup when the match is >= {L} tokens: {lookup_tok / out_tokens:6.1%} of output in "
              f"{rounds:,} lookup rounds, {acc / max(rounds, 1):.2f} drafts accepted per round; "
              f"{t / 1000:.1f} s, {model_ms / t:.2f}x the model drafter alone")


if __name__ == "__main__":
    sys.exit(main())
