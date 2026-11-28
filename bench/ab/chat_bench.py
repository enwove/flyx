# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Prompt sets through an OpenAI-compatible /v1/chat/completions server (`flyx serve`, SGLang, ...) at a concurrency
and a temperature: the same requests for every engine.  Standard library only.

    python -I bench/ab/chat_bench.py --url http://127.0.0.1:8080 [--conc N] [--temp T] [--top-p P] [--seed S]
        [--warm 1] [--max-new 256] [--json OUT.json] SET.jsonl ...

A set is JSON lines with a "question" each, sent as one user message, thinking off (chat_template_kwargs
enable_thinking false), presence and frequency penalties 0, streamed; request i of a set has seed --seed + i.
Per request: new tokens (the server's usage), time to the first content, decode time (first content to the end of the
stream).  Per set and over all: decode tok/s (tokens after the first over the decode times, the per-request speed)
and the throughput (every new token over the wall time of the timed pass).  --warm 1 runs every request once untimed
first.
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import urllib.request


def chat_one(url, model, question, max_new, temp, top_p, top_k, seed):
    body = {"model": model, "messages": [{"role": "user", "content": question}], "max_tokens": max_new,
            "temperature": temp, "top_p": top_p, "presence_penalty": 0, "frequency_penalty": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
            "seed": seed}
    if top_k > 0:
        body["top_k"] = top_k
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    usage = {}
    text = []
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            j = json.loads(payload)
            usage = j.get("usage") or usage
            ch = j.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content"):
                text.append(ch[0]["delta"]["content"])
                if first is None:
                    first = time.perf_counter()
    end = time.perf_counter()
    return {"new": usage.get("completion_tokens", 0), "prompt_tokens": usage.get("prompt_tokens", 0),
            "ttft": (first or end) - t0, "decode_s": end - (first or end), "text": "".join(text), "t0": t0,
            "end": end}


def run_batch(fn, jobs, conc):
    """jobs: (key, args); every job through fn on `conc` threads.  Returns ({key: result}, wall seconds, errors)."""
    q: queue.Queue = queue.Queue()
    for j in jobs:
        q.put(j)
    out, errs = {}, []

    def worker():
        while True:
            try:
                key, args = q.get_nowait()
            except queue.Empty:
                return
            try:
                out[key] = fn(*args)
            except Exception as e:  # noqa: BLE001
                errs.append(f"{key}: {e}")

    t0 = time.perf_counter()
    ts = [threading.Thread(target=worker) for _ in range(conc)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0
    for e in errs[:5]:
        print("error", e, file=sys.stderr)
    return out, wall, len(errs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model-name", default="m", help="the requests' model field")
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--json")
    a = ap.parse_args(argv)

    sets = [(path, [json.loads(line)["question"] for line in open(path, encoding="utf-8") if line.strip()])
            for path in a.sets]
    jobs = [((si, qi), (q, a.seed + qi)) for si, (_, qs) in enumerate(sets) for qi, q in enumerate(qs)]

    def fn(question, seed):
        return chat_one(a.url, a.model_name, question, a.max_new, a.temp, a.top_p, a.top_k, seed)

    if a.warm:
        run_batch(fn, jobs, a.conc)
    res, wall, nerr = run_batch(fn, jobs, a.conc)

    def summary(keys):
        rows = [res[k] for k in keys if k in res]
        tok = sum(r["new"] - 1 for r in rows if r["new"] > 0)
        dec = sum(r["decode_s"] for r in rows)
        return {"n": len(rows), "tokens": sum(r["new"] for r in rows), "decode_tok_s": tok / dec if dec else 0,
                "ttft_ms": 1000 * sum(r["ttft"] for r in rows) / max(1, len(rows)),
                "prompt_tokens": sum(r["prompt_tokens"] for r in rows)}

    report = {"url": a.url, "conc": a.conc, "temp": a.temp, "top_p": a.top_p, "top_k": a.top_k, "seed": a.seed,
              "errors": nerr, "wall_s": wall, "sets": {}}
    for si, (path, _) in enumerate(sets):
        name = path.rsplit("/", 1)[-1].removesuffix(".jsonl")
        report["sets"][name] = summary([k for k, _ in jobs if k[0] == si])
    keys = [k for k, _ in jobs]
    report["all"] = summary(keys)
    report["all"]["throughput_tok_s"] = sum(res[k]["new"] for k in keys if k in res) / wall
    for name, s in report["sets"].items():
        print(f"{name:12s} {s['decode_tok_s']:7.1f} tok/s  ttft {s['ttft_ms']:6.1f} ms")
    s = report["all"]
    print(f"{'all':12s} {s['decode_tok_s']:7.1f} tok/s  throughput {s['throughput_tok_s']:7.1f} tok/s  "
          f"wall {wall:.1f}s  errors {nerr}")
    if a.json:
        report["replies"] = [{"key": list(k), **{x: y for x, y in res[k].items() if x not in ("t0", "end")}}
                             for k, _ in jobs if k in res]
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
