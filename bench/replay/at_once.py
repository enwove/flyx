# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Requests sent at once, against an OpenAI-compatible server: each one's speed and all of them together.

    python bench/replay/at_once.py --url http://127.0.0.1:18080 [--n 1,2,4] [--tokens 256] [--runs 2]
                                   [--questions FILE.jsonl]

For each count in --n, that many streamed chat requests start together (different questions, greedy, thinking
off, --tokens each); reported per count, medians over --runs: the requests' own tokens per second (from their first
token to their last), and all of them together (their tokens over the time from the first request sent to the last
token received).  A server whose requests take turns on the GPU gives about one request's total whatever the count;
one whose rounds hold several requests' rows (flyx) more. --questions: the requests' questions from a
prompt set as `flyx bench --prompts` reads it (a {"question": ...} a line), request i the set's i-th; else the ones
below.  Standard library only.
"""
from __future__ import annotations

import argparse
import statistics
import threading
import time
import urllib.request

from http_api import api_request, chat_events

QUESTIONS = [
    "Write a Python function that merges two sorted lists, with tests.",
    "Explain how a hash map handles collisions, with an example.",
    "Describe the water cycle for a ten-year-old.",
    "Write a short story about a robot that learns to paint.",
    "Compare TCP and UDP and say when to use each.",
    "Explain what a closure is in JavaScript, with code.",
    "List the planets of the solar system with one fact each.",
    "Write a SQL query that finds the second highest salary, and explain it.",
]


def one(url, question, tokens, out, i):
    body = {"model": "m", "messages": [{"role": "user", "content": question}], "max_tokens": tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    first = last = None
    n = 0
    with urllib.request.urlopen(api_request(url, "/v1/chat/completions", body), timeout=600) as resp:
        for chunk in chat_events(resp):
            for c in chunk.get("choices", []):
                if (c.get("delta") or {}).get("content"):
                    last = time.perf_counter()
                    first = first or last
            if chunk.get("usage"):
                n = chunk["usage"].get("completion_tokens", 0)
    out[i] = (first, last, n)


def run(url, n, tokens, questions=QUESTIONS):
    out = [None] * n
    threads = [threading.Thread(target=one, args=(url, questions[i % len(questions)], tokens, out, i))
               for i in range(n)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    own = [(k - 1) / (b - a) for a, b, k in out if a and b and b > a and k > 1]
    total = sum(k for _, _, k in out) / (max(b for _, b, _ in out) - t0)
    return statistics.median(own), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18080")
    ap.add_argument("--n", default="1,2,4")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--questions", help="a prompt set, a JSON object a line with its question")
    a = ap.parse_args()
    questions = QUESTIONS
    if a.questions:
        with open(a.questions, encoding="utf-8") as f:
            questions = [json.loads(line)["question"] for line in f if line.strip()]
    run(a.url, 1, 16, questions)  # warm
    print(f"{'at once':>7} {'each tok/s':>11} {'all tok/s':>10}")
    for n in (int(x) for x in a.n.split(",")):
        r = [run(a.url, n, a.tokens, questions) for _ in range(a.runs)]
        print(f"{n:>7} {statistics.median(x for x, _ in r):11.1f} {statistics.median(y for _, y in r):10.1f}",
              flush=True)


if __name__ == "__main__":
    main()
