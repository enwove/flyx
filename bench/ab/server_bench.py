# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The prompts of a `flyx bench --prompts --json` record, token for token, through another engine's server (SGLang,
vLLM or llama.cpp): its decode speed as flyx bench counts it, its tokens a round where it reports them, and how many
of its answers are the record's.

    python -I bench/ab/server_bench.py RECORD.json --api sglang|vllm|llamacpp [--url URL] [--model NAME]
        [--json OUT.json] [--same-as RUN.json]

Each prompt's token ids (the record's prompt_ids, the chat template as flyx rendered it) go to the server one request
at a time, streamed, greedy, with the record's end tokens and token limit: SGLang's /generate (input_ids), vLLM's
/v1/completions (prompt as ids, stop_token_ids, return_token_ids), llama.cpp's /completion (prompt as ids,
return_tokens, its end-of-generation tokens).  Decode tok/s: the tokens after the first over the time from the first
token to the last, summed over the prompts (flyx bench's decode_tok_s); prefill: the time to the first token.  Tokens
a round: SGLang's verification passes (meta_info.spec_verify_ct; none without speculation).  Same: the replies whose
tokens are the record's answer; with --same-as, those that are another run's (an earlier --json of the same record).
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

DEFAULT_URL = {"sglang": "http://127.0.0.1:30000", "vllm": "http://127.0.0.1:8000", "llamacpp": "http://127.0.0.1:8080"}


def stream(url, body):
    """The server's streamed events (data: lines) as JSON, with the time each arrived."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                return
            yield time.perf_counter(), json.loads(payload)


def generate(api, url, model, ids, eos, max_new):
    start = time.perf_counter()
    first = None
    out, verify, finish = [], 0, None
    if api == "sglang":
        body = {"input_ids": ids, "stream": True,
                "sampling_params": {"temperature": 0, "max_new_tokens": max_new, "stop_token_ids": eos}}
        last = None
        for t, j in stream(url + "/generate", body):
            if first is None and j.get("meta_info", {}).get("completion_tokens", 0) >= 1:
                first = t
            last = j
        m = last["meta_info"]
        out, verify, finish = last.get("output_ids") or [], m.get("spec_verify_ct", 0), m.get("finish_reason")
        new = m["completion_tokens"]
    elif api == "vllm":
        body = {"model": model, "prompt": ids, "max_tokens": max_new, "temperature": 0, "stream": True,
                "stop_token_ids": eos, "skip_special_tokens": False, "return_token_ids": True,
                "stream_options": {"include_usage": True}}
        new = 0
        for t, j in stream(url + "/v1/completions", body):
            for c in j.get("choices", []):
                got = c.get("token_ids") or []
                if got and first is None:
                    first = t
                out += got
                finish = c.get("finish_reason") or finish
            if j.get("usage"):
                new = j["usage"]["completion_tokens"]
    else:  # llama.cpp
        body = {"prompt": ids, "n_predict": max_new, "temperature": 0, "top_k": 1, "stream": True,
                "cache_prompt": False, "return_tokens": True}
        for t, j in stream(url + "/completion", body):
            got = j.get("tokens") or []
            if got and first is None:
                first = t
            out += got
            if j.get("stop"):
                finish = j.get("stop_type")
        new = len(out)
    end = time.perf_counter()
    return {"new": new, "prefill_s": (first or end) - start, "decode_s": end - (first or end), "verify": verify,
            "ids": out, "finish": finish}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("record", help="a flyx bench --prompts --json record (its prompt_ids, end tokens and limit)")
    ap.add_argument("--api", required=True, choices=sorted(DEFAULT_URL))
    ap.add_argument("--url")
    ap.add_argument("--model", default="", help="vLLM's served model name")
    ap.add_argument("--json", help="every reply's numbers and tokens")
    ap.add_argument("--same-as", help="an earlier --json of this record: how many replies are its")
    a = ap.parse_args(argv)
    url = a.url or DEFAULT_URL[a.api]
    rec = json.load(open(a.record, encoding="utf-8"))
    other = json.load(open(a.same_as, encoding="utf-8"))["replies"] if a.same_as else None
    eos, max_new = rec["eos_ids"], rec["max_new_tokens"]
    rows = []
    for p in rec["prompts"]:
        r = generate(a.api, url, a.model, p["prompt_ids"], eos, max_new)
        r["same"] = r["ids"] == p["ids"]
        rows.append(r)
        print(f"prompt {len(rows):3d}: {r['new']:5d} tokens, {r['decode_s'] * 1e3:8.1f} ms after the first"
              f"{', same' if r['same'] else ''}", file=sys.stderr)
    tok = sum(max(0, r["new"] - 1) for r in rows)
    sec = sum(r["decode_s"] for r in rows)
    verify = sum(r["verify"] for r in rows)
    out = {"record": a.record, "api": a.api, "url": url, "prompts": len(rows), "decode_tok_s": tok / sec if sec else 0.0,
           "prefill_s": sum(r["prefill_s"] for r in rows), "tokens_per_round": tok / verify if verify else None,
           "same_as_record": sum(r["same"] for r in rows), "record_decode_tok_s": rec.get("decode_tok_s")}
    if other is not None:
        out["same_as_run"] = sum(r["ids"] == o["ids"] for r, o in zip(rows, other))
    print(json.dumps(out))
    if a.json:
        json.dump({**out, "replies": rows}, open(a.json, "w", encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
