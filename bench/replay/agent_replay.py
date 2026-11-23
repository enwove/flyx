# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Replay a multi-turn agent session against an OpenAI-compatible server.

    python bench/replay/agent_replay.py --url http://127.0.0.1:18080 [--turns 8] [--runs 3] [--calls N]

Each turn appends the model's reply (with its tool calls) and a canned tool
result, the way coding agents do, and asks again.  --calls N makes the first N
turns after each question call a tool (tool_choice "required"), as long agent
loops do; --system-tokens N pads the system prompt as agents' tool
documentation does.  Reported per turn: whether it answers a user message or a tool
result, prompt tokens, tokens recomputed (prompt - cached), time to the first
streamed token, decode speed; and over the session: p50 / p95 time to first
token for each kind of turn and the total time.  Works against any server that reports
usage.prompt_tokens_details.cached_tokens (flyx, vLLM, ...). Set FLYX_KEY if the server requires an API key.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from http_api import api_request, chat_events

SYSTEM = ("You are a coding agent working in a Python repository. Use the tools to inspect files before "
          "answering. Keep answers short.")
TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file of the repository.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                     "required": ["path"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search the repository for a pattern.",
                                      "parameters": {"type": "object", "properties": {
                                          "pattern": {"type": "string"}, "path": {"type": "string"}},
                                          "required": ["pattern"]}}},
]
QUESTIONS = [
    "Where is the HTTP retry logic implemented?",
    "Does it back off exponentially? Check the code.",
    "Add jitter to the backoff. Show the new function.",
    "Now write a unit test for it.",
]
FILE = "\n".join(f"def helper_{i}(x):\n    return x * {i} + {i % 7}\n" for i in range(40))


def tool_result(name: str, args: dict) -> str:
    if name == "read_file":
        return f"# {args.get('path', '?')}\nimport random, time\n\n" + FILE + \
            "\ndef retry(fn, attempts=5, base=0.5):\n    for i in range(attempts):\n        try:\n" \
            "            return fn()\n        except IOError:\n            time.sleep(base * 2 ** i)\n" \
            "    raise RuntimeError('gave up')\n"
    return "src/net/http.py:88: def retry(fn, attempts=5, base=0.5):\nsrc/net/http.py:93:  time.sleep(base * 2 ** i)"


def chat_stream(url: str, body: dict):
    request = api_request(url, "/v1/chat/completions", body)
    first = None
    content, reasoning, calls, usage = "", "", {}, None
    t0 = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        for event in chat_events(response):
            if event.get("usage"):
                usage = event["usage"]
            for ch in event.get("choices", []):
                delta = ch.get("delta", {})
                if first is None and (delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")):
                    first = time.perf_counter()
                content += delta.get("content") or ""
                reasoning += delta.get("reasoning_content") or ""
                for tc in delta.get("tool_calls") or []:
                    call = calls.setdefault(tc["index"], {"id": "", "function": {"name": "", "arguments": ""}})
                    call["id"] += tc.get("id") or ""
                    for key in ("name", "arguments"):
                        call["function"][key] += (tc.get("function") or {}).get(key) or ""
        end = time.perf_counter()
    return content, reasoning, [calls[k] for k in sorted(calls)], usage, (first if first is not None else end) - t0, end - t0


def system_prompt(tokens: int) -> str:
    """The system prompt, padded with tool documentation to about `tokens` tokens (agents send 10-30K)."""
    lines = [SYSTEM]
    size = len(SYSTEM) + 1
    while size < tokens * 4:  # ~4 bytes a token
        i = len(lines) - 1
        note = (f"Tool note {i}: run_task_{i} takes a path and a pattern; it returns the matching lines "
                f"of the file, at most {i % 50 + 10} of them, and fails on a missing file.")
        lines.append(note)
        size += len(note) + 1
    return "\n".join(lines)


def run_session(url: str, turns: int, think: bool, forced_calls: int = 0, system_tokens: int = 0):
    msgs = [{"role": "system", "content": system_prompt(system_tokens)}]
    rows = []
    q = 0
    since_question = 0
    for turn in range(turns):
        kind = "tool"
        if len(msgs) == 1 or msgs[-1]["role"] == "assistant":
            msgs.append({"role": "user", "content": QUESTIONS[q % len(QUESTIONS)]})
            q += 1
            kind, since_question = "user", 0
        body = {"messages": msgs, "tools": TOOLS, "stream": True, "stream_options": {"include_usage": True},
                "max_tokens": 512, "temperature": 0, "chat_template_kwargs": {"enable_thinking": think}}
        if since_question < forced_calls:
            body["tool_choice"] = "required"
        since_question += 1
        content, reasoning, calls, usage, ttft, total = chat_stream(url, body)
        prompt = usage["prompt_tokens"] if usage else 0
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) if usage else 0
        out = usage["completion_tokens"] if usage else 0
        rows.append({"turn": turn, "kind": kind, "prompt": prompt, "recomputed": prompt - cached, "ttft": ttft, "total": total,
                     "tok_s": (out - 1) / max(total - ttft, 1e-6) if out > 1 else 0.0, "calls": len(calls)})
        msg = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if calls:
            msg["tool_calls"] = [{"id": c["id"], "type": "function", "function": c["function"]} for c in calls]
        msgs.append(msg)
        for c in calls:
            args = json.loads(c["function"].get("arguments") or "{}")
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": tool_result(c["function"]["name"], args)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18080")
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--think", action="store_true", help="let the model think before answering")
    ap.add_argument("--calls", type=int, default=0, help="turns after each question that must call a tool")
    ap.add_argument("--system-tokens", type=int, default=0, help="pad the system prompt to about this many tokens")
    a = ap.parse_args()
    ttfts, totals, recomputed = {"user": [], "tool": []}, [], {"user": [], "tool": []}
    for run in range(a.runs):
        t0 = time.perf_counter()
        rows = run_session(a.url, a.turns, a.think, a.calls, a.system_tokens)
        totals.append(time.perf_counter() - t0)
        print(f"run {run}: {'turn':>4} {'kind':>4} {'prompt':>7} {'recomp':>7} {'ttft ms':>8} {'tok/s':>7} calls")
        for r in rows:
            print(f"        {r['turn']:>4} {r['kind']:>4} {r['prompt']:>7} {r['recomputed']:>7} "
                  f"{r['ttft'] * 1000:>8.0f} {r['tok_s']:>7.1f} {r['calls']:>5}")
            ttfts[r["kind"]].append(r["ttft"])
            recomputed[r["kind"]].append(r["recomputed"])
    for kind in ("user", "tool"):
        t = ttfts[kind]
        if not t:
            continue
        q = statistics.quantiles(t, n=20) if len(t) > 1 else [t[0]] * 19
        print(f"{kind} turns ({len(t)}): time to first token p50 {statistics.median(t) * 1000:.0f} ms, "
              f"p95 {q[18] * 1000:.0f} ms; recomputed median {statistics.median(recomputed[kind]):.0f}, "
              f"max {max(recomputed[kind])} tokens")
    print(f"session: median {statistics.median(totals):.1f} s ({a.turns} turns)")


if __name__ == "__main__":
    main()
