# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Conversations that take turns on fewer sessions than there are of them, against flyx-server.

    python bench/replay/take_turns.py --url http://127.0.0.1:18080 [--conversations 3] [--turns 4]
        [--context bench/prompts/long.txt] [--max-tokens 32]

Each conversation opens with a long document (the context file under a heading of its own, so no two share a
prefix) and asks about it, turn after turn; they take turns, one request at a time.  With fewer sessions than
conversations (flyx-server --sessions 2), every turn after a conversation's first finds its session taken by
another: the server computes the whole prompt again, or loads the conversation's checkpoint from host memory
(--host-cache).  Reported per request: the time to the first token, the prompt's tokens and how many the server
reused (usage.prompt_tokens_details.cached_tokens); then the medians of the first turns and of the returning ones.
Greedy, thinking off; standard library only.
"""
from __future__ import annotations

import argparse
import statistics
import time
import urllib.request

from http_api import api_request, chat_events

QUESTIONS = [
    "In one sentence: what is this document about?",
    "Name three details it mentions.",
    "Which part of it would a newcomer find hardest?",
    "Give it a short title.",
    "What would you ask its author?",
    "Summarize it in two sentences.",
]


def ask(url, messages, max_tokens):
    """One streamed chat request: (time to first token in s, the reply's text, prompt tokens, reused tokens)."""
    body = {"model": "m", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter()
    first, text, usage = None, [], {}
    with urllib.request.urlopen(api_request(url, "/v1/chat/completions", body), timeout=600) as resp:
        for chunk in chat_events(resp):
            for c in chunk.get("choices", []):
                piece = (c.get("delta") or {}).get("content")
                if piece:
                    if first is None:
                        first = time.perf_counter() - t0
                    text.append(piece)
            if chunk.get("usage"):
                usage = chunk["usage"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    return first, "".join(text), usage.get("prompt_tokens", 0), cached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18080")
    ap.add_argument("--conversations", type=int, default=3)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--context", default="bench/prompts/long.txt")
    ap.add_argument("--max-tokens", type=int, default=32)
    a = ap.parse_args()
    with open(a.context, encoding="utf-8") as f:
        doc = f.read()
    convs = [[{"role": "system", "content": f"Document {i + 1} of {a.conversations}, kept apart from the others.\n\n"
               + doc}] for i in range(a.conversations)]
    first_turns, returns = [], []
    print(f"{'turn':>4} {'conv':>4} {'ttft s':>8} {'prompt':>7} {'reused':>7}")
    for t in range(a.turns):
        for i, msgs in enumerate(convs):
            msgs.append({"role": "user", "content": QUESTIONS[t % len(QUESTIONS)]})
            ttft, reply, prompt, cached = ask(a.url, msgs, a.max_tokens)
            msgs.append({"role": "assistant", "content": reply})
            (first_turns if t == 0 else returns).append(ttft)
            print(f"{t + 1:>4} {i + 1:>4} {ttft:8.3f} {prompt:>7} {cached:>7}", flush=True)
    print(f"median time to the first token: first turns {statistics.median(first_turns):.3f} s, "
          f"returning turns {statistics.median(returns):.3f} s ({len(returns)} requests)")


if __name__ == "__main__":
    main()
