# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Time an Anthropic prompt prewarm and its first generated token. Set FLYX_KEY for an authenticated server."""
import argparse
import json
import time
import urllib.request

from claude_cache import user
from http_api import api_request, get


def tools(prefix, count):
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "active": {"type": "boolean"},
              "status": {"type": "string", "enum": ["queued", "ready", "finished"]}},
              "required": ["name"], "additionalProperties": False}
    return [{"name": f"{prefix}_{i:03}", "description": "Find a record in the specified state.", "input_schema": schema}
            for i in range(count)]


def measure(url, model, definitions, limit, prompt):
    body = {"model": model, "messages": [user(prompt)], "tools": definitions, "max_tokens": limit,
            "temperature": 0, "thinking": {"type": "disabled"}}
    req = api_request(url, "/v1/messages", body)
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as response:
        message = json.load(response)
    elapsed = (time.perf_counter() - start) * 1000
    if message["usage"]["output_tokens"] != limit or (limit == 0 and message["content"]):
        raise RuntimeError(f"unexpected prewarm or generation result: {message}")
    return elapsed, message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    model = get(args.url, "/v1/models")["data"][0]["id"]
    measure(args.url, model, [], 1, "Initial request.")
    for repeat in range(args.repeats):
        for count, same in ((8, True), (32, False), (8, False), (32, True)):
            suffix = f"{repeat}{count:02}{int(same)}"
            first = tools("find" + suffix, count)
            following = first if same else tools("read" + suffix, count)
            prompt = "Are the records ready?"
            warm_ms, warm = measure(args.url, model, first, 0, prompt)
            next_ms, reply = measure(args.url, model, following, 1, prompt)
            print(json.dumps({"repeat": repeat, "tools": count, "same": same,
                              "prewarm_ms": round(warm_ms, 3), "next_ms": round(next_ms, 3),
                              "total_ms": round(warm_ms + next_ms, 3), "prewarm_usage": warm["usage"],
                              "next_usage": reply["usage"], "output": reply["content"]}), flush=True)


if __name__ == "__main__":
    main()
