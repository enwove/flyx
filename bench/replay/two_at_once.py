# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Two requests that overlap, against an OpenAI-compatible server.

    python bench/replay/two_at_once.py --url http://127.0.0.1:18080 [--runs 3] [--first 512] [--second 128]

A long reply is being written when a second, unrelated request arrives (sent
once the first has streamed 32 pieces).  Reported per run, and as medians:

  alone     each request by itself: how long it takes, tokens per second
  together  the first request's time; the second's time to its first token
            and its time, both from the moment it was sent; the time from the
            first request's start until both have ended

A server that runs one request at a time makes the second wait for the first
to end: its time to first token is what is left of the first one.  A server
that lets them take turns starts it at once, and both run slower while they
overlap. The report also counts runs where the first reply was still active
when the second was sent. Set FLYX_KEY if the server requires an API key.
Greedy, thinking off; standard library only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import statistics
import threading
import time
import urllib.request
from http_api import api_request, chat_events, get

FIRST = "Write a long, detailed story about a lighthouse keeper who repairs clocks."
SECOND = "Explain in a few short paragraphs how a bicycle stays upright."


def stream(url: str, body: dict, on_piece=None) -> dict:
    """One streamed chat completion: when it started, its first piece, its end, and its tokens."""
    request = api_request(url, "/v1/chat/completions", body)
    out = {"sent": time.perf_counter(), "first": None, "tokens": 0, "pieces": 0}
    with urllib.request.urlopen(request, timeout=600) as response:
        for event in chat_events(response):
            if event.get("usage"):
                out["tokens"] = event["usage"]["completion_tokens"]
            if any(ch["delta"].get("content") for ch in event.get("choices", [])):
                out["pieces"] += 1
                if out["first"] is None:
                    out["first"] = time.perf_counter()
                if on_piece:
                    on_piece(out["pieces"])
    out["end"] = time.perf_counter()
    if out["first"] is None:
        raise RuntimeError("the server returned no visible text to time")
    return out


def body(model: str, prompt: str, n: int) -> dict:
    return {"model": model, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": n,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}


def speed(r: dict) -> float:
    return (r["tokens"] - 1) / (r["end"] - r["first"]) if r["tokens"] > 1 and r["end"] > r["first"] else 0.0


def concurrent_pair(url, first, second):
    writing = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as worker:
        foreground = worker.submit(stream, url, first, lambda pieces: pieces >= 32 and writing.set())
        foreground.add_done_callback(lambda _: writing.set())
        writing.wait()
        if foreground.done():
            foreground.result()  # Propagate a failed first request before starting another.
        reply = stream(url, second)
        return foreground.result(), reply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--first", type=int, default=512, help="tokens of the first reply")
    ap.add_argument("--second", type=int, default=128, help="tokens of the second reply")
    a = ap.parse_args()
    if min(a.runs, a.first, a.second) < 1:
        ap.error("--runs, --first and --second must all be positive")
    model = get(a.url, "/v1/models")["data"][0]["id"]
    first, second = body(model, FIRST, a.first), body(model, SECOND, a.second)
    stream(a.url, second)  # warm: both prompts seen once, so every run below reuses its session alike
    stream(a.url, first)

    rows = []
    for run in range(a.runs):
        alone_1, alone_2 = stream(a.url, first), stream(a.url, second)
        f, s = concurrent_pair(a.url, first, second)
        row = {
            "alone first s": alone_1["end"] - alone_1["sent"], "alone first tok/s": speed(alone_1),
            "alone second s": alone_2["end"] - alone_2["sent"], "alone second tok/s": speed(alone_2),
            "first s": f["end"] - f["sent"], "first tok/s": speed(f),
            "second to first token s": s["first"] - s["sent"], "second s": s["end"] - s["sent"],
            "second tok/s": speed(s), "both done s": max(f["end"], s["end"]) - f["sent"],
            "second ended before first": s["end"] < f["end"],
            "second sent during first": s["sent"] < f["end"],
        }
        rows.append(row)
        print(f"run {run}: " + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
    print(f"\nmedian of {len(rows)} runs ({a.first} + {a.second} tokens):")
    for k in rows[0]:
        vals = [r[k] for r in rows]
        if isinstance(vals[0], bool):
            print(f"  {k:28s} {sum(vals)} of {len(vals)}")
        else:
            print(f"  {k:28s} {statistics.median(vals):8.2f}")


if __name__ == "__main__":
    main()
