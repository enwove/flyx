# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Replay a synthetic Claude Code conversation, a permission check and an early history edit.

    python bench/replay/claude_cache.py --url http://127.0.0.1:18085 [--disk-dir DIR] [--compare reference.jsonl]

Run each variant against a fresh server. Report client time to first content, engine
prefill and cache hits, and a digest of each reply. --disk-dir reports the total length of live snapshot
files. --case decode begins just below 10,240 tokens and crosses the boundary during sampled generation.
--case shared-system sends a title alongside a 24K conversation with identical system instructions;
use two sessions to measure reuse across slots. title-during-decode sends the title as soon as the
main reply starts streaming. title-after-decode waits for that reply to finish. These two cases use
the usage on each stream and measure the longest gap between visible text events.
For a server started with --api-key, put its key in FLYX_KEY.
"""
from __future__ import annotations

import argparse
import codecs
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import pathlib
import threading
import time
import urllib.request
from http_api import api_request, get


def snapshot_bytes(directory):
    total = 0
    for file in directory.rglob("*"):
        try:
            if file.is_file():
                total += file.stat().st_size
        except FileNotFoundError:
            pass  # an async writer can publish its temporary file during this measurement
    return total


def ask(url, model, system, messages, tools, disk_dir, max_tokens, temperature, *, timeline=None, on_first=None):
    body = {"model": model, "system": system, "messages": messages, "tools": tools, "stream": True,
            "thinking": {"type": "disabled"}, "max_tokens": max_tokens, "temperature": temperature, "seed": 42}
    request = api_request(url, "/v1/messages", body)
    start = time.perf_counter()
    first = None
    previous = None
    longest_gap = 0.0
    usage = {}
    reply = []
    stop_reason = None
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            event = json.loads(line[5:])
            if event["type"] == "error":
                raise RuntimeError(event["error"])
            if timeline is not None and event["type"] == "message_start":
                usage.update(event["message"].get("usage", {}))
            if event["type"] == "content_block_delta":
                delta = event["delta"]
                if delta["type"] != "text_delta":
                    raise RuntimeError(f"unexpected response block: {delta['type']}")
                if delta["text"]:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                        if on_first:
                            on_first()
                    else:
                        longest_gap = max(longest_gap, now - previous)
                    previous = now
                reply.append(delta["text"])
            if event["type"] == "message_delta":
                stop_reason = event["delta"]["stop_reason"]
                if timeline is not None:
                    usage.update(event.get("usage", {}))
    end = time.perf_counter()
    if first is None or stop_reason is None:
        raise RuntimeError("incomplete reply")
    text = "".join(reply)
    if timeline is None:
        turn = max(get(url, "/v1/telemetry")["recent"], key=lambda row: row["finished_at"])
        row = {key: turn[key] for key in ("session", "prompt_tokens", "reused_tokens", "output_tokens", "prefill_ms", "queue_ms")}
    else:
        # The latest telemetry entry can belong to the other stream.
        reused = usage.get("cache_read_input_tokens", 0)
        row = {"prompt_tokens": usage["input_tokens"] + usage.get("cache_creation_input_tokens", 0) + reused,
               "reused_tokens": reused, "output_tokens": usage["output_tokens"],
               "start_s": round(start - timeline, 3), "max_content_gap_s": round(longest_gap, 3)}
    row.update(ttft_s=round(first - start, 3), elapsed_s=round(end - start, 3),
               reply_sha256=hashlib.sha256(text.encode()).hexdigest(), stop=stop_reason)
    if disk_dir:
        row["disk_bytes"] = snapshot_bytes(disk_dir)
    return text, row


def user(content):
    return {"role": "user", "content": content}


def assistant(content):
    return {"role": "assistant", "content": content}


def fill(n):
    return " red blue green white" * (n // 4)


def shared_system_turns(request, title_timing=None):
    system = [{"type": "text", "text": "You track a synthetic inventory.\n" + fill(16000),
               "cache_control": {"type": "ephemeral"}}]
    messages = [user("Inventory batch A.\n" + fill(4000)), assistant("Recorded."),
                user("Inventory batch B.\n" + fill(4000) + "\nRespond with BLUE.")]
    reply, _ = request("main", system, messages, [])
    title = ("title", system, [user("A conversation about an inventory. Return one word.")], [])
    if title_timing:
        messages.extend([assistant(reply), user("List the integers from 1 to 500, each on its own line. Continue until done.")])
        working = ("working", system, messages, [], 160)
        if title_timing == "title-after-decode":
            request(*working)
            request(*title)
        else:
            with ThreadPoolExecutor(max_workers=1) as worker:
                future = None

                def start_title():
                    nonlocal future
                    future = worker.submit(request, *title)

                request(*working, on_first=start_title)
                future.result()
        return
    request(*title)
    messages.extend([assistant(reply), user("Answer with GREEN.")])
    request("return", system, messages, [])


def agent_turns(request):
    marker = {"type": "ephemeral"}
    system = [{"type": "text", "text": "You summarize a synthetic inventory.\n" + fill(16000),
               "cache_control": marker}]
    tools = [{"name": "lookup", "description": "Look up an inventory label.",
              "input_schema": {"type": "object", "properties": {"label": {"type": "string"}}}}]
    messages = [user("Here is the first list.\n" + fill(15000)), assistant("Received."),
                user("Look up item blue."),
                assistant([{"type": "tool_use", "id": "inv1", "name": "lookup", "input": {"label": "blue"}}])]
    result = {"type": "tool_result", "tool_use_id": "inv1", "content": "Blue is available."}
    messages.append(user([result, {"type": "text", "text": "Inventory notes.\n" + fill(19000)}]))
    messages.extend([assistant("Noted."), user("Second list.\n" + fill(19000)), assistant("Noted."),
                     user("Final list.\n" + fill(18700) + "\nAnswer with just BLUE.")])
    reply, _ = request("main", system, messages, tools)
    classifier = "Classify synthetic inventory requests.\n" + fill(41500)
    request("aux", classifier, [user("May the inventory be read? Answer with just YES.")], [])
    messages.extend([assistant(reply), user("Answer with just GREEN.")])
    reply, _ = request("return", system, messages, tools)
    # A short reminder is added to an early tool result, with the same ID, before the summary request.
    result["content"] += "\n<system-reminder>The inventory is synthetic.</system-reminder>"
    messages.extend([assistant(reply), user("Additional notes.\n" + fill(16000) + "\nSummarize in three words.")])
    request("edited", system, messages, tools)


def decode_turns(url, model, request):
    system = "You follow the user's formatting instructions."
    question = "\nPrint the integers from 1 to 150 in order, one number per line."

    def history(n):
        return [user("Ignore this padding." + " red" * n + question)]

    body = {"model": model, "system": system, "messages": history(0), "thinking": {"type": "disabled"}}
    count = api_request(url, "/v1/messages/count_tokens", body)
    with urllib.request.urlopen(count, timeout=30) as response:
        pad = 10236 - json.load(response)["input_tokens"]
    messages = history(pad)
    reply, row = request("decode", system, messages, [], 80, 0.7)
    if not row["prompt_tokens"] < 10240 < row["prompt_tokens"] + row["output_tokens"] - 1:
        raise RuntimeError("the reply did not cross the staging boundary")
    # This normal prefill creates a restorable anchor for the staged decode prefix.
    messages.extend([assistant(reply), user("Give the next three numbers.")])
    reply, _ = request("decode_followup", system, messages, [], 16, 0.7)
    request("decode_aux", "Classify this item.", [user("Reply with just YES.")], [])
    messages.extend([assistant(reply), user("Give the next two numbers.")])
    request("decode_return", system, messages, [], 16, 0.7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18085")
    parser.add_argument("--disk-dir", type=pathlib.Path)
    parser.add_argument("--case", choices=("agent", "decode", "shared-system", "title-during-decode", "title-after-decode"),
                        default="agent")
    parser.add_argument("--compare", type=pathlib.Path, help="check replies against a previous run of this case (JSONL)")
    args = parser.parse_args()
    expected = {}
    if args.compare:
        data = args.compare.read_bytes()
        # Native redirection in Windows PowerShell 5.1 writes UTF-16; newer shells usually write UTF-8.
        encoding = "utf-16" if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)) else "utf-8-sig"
        for line_number, line in enumerate(data.decode(encoding).splitlines(), start=1):
            row = json.loads(line)
            name = row["turn"]
            if name in expected:
                parser.error(f"{args.compare}:{line_number}: duplicate reference turn {name!r}")
            expected[name] = row

    model = get(args.url, "/v1/models")["data"][0]["id"]
    title_timing = args.case if args.case in ("title-during-decode", "title-after-decode") else None
    timeline = time.perf_counter() if title_timing else None
    reporting = threading.Lock()

    def request(name, system, messages, tools, max_tokens=12, temperature=0, on_first=None):
        reply, row = ask(args.url, model, system, messages, tools, args.disk_dir, max_tokens, temperature,
                         timeline=timeline, on_first=on_first)
        with reporting:
            print(json.dumps({"turn": name, **row}), flush=True)
            if args.compare:
                want = expected.pop(name)
                for key in ("prompt_tokens", "output_tokens", "reply_sha256", "stop"):
                    if row[key] != want[key]:
                        raise RuntimeError(f"{name} {key}: got {row[key]}, reference {want[key]}")
        return reply, row

    if args.case == "decode":
        decode_turns(args.url, model, request)
    elif args.case == "shared-system":
        shared_system_turns(request)
    elif title_timing:
        shared_system_turns(request, title_timing)
    else:
        agent_turns(request)
    if expected:
        raise RuntimeError(f"reference has additional turns: {sorted(expected)}")


if __name__ == "__main__":
    main()
