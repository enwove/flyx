# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""HTTP requests shared by the local server replays."""
import json
import os
import urllib.request


def api_request(url, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url.rstrip("/") + path, data=data)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    if key := os.environ.get("FLYX_KEY"):
        request.add_unredirected_header("Authorization", "Bearer " + key)
    return request


def get(url, path):
    with urllib.request.urlopen(api_request(url, path), timeout=600) as response:
        return json.load(response)


def chat_events(response):
    finished = False
    for line in response:
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            return
        event = json.loads(data)
        if "error" in event:
            raise RuntimeError(f"chat stream failed: {event['error']}")
        if any(choice.get("finish_reason") for choice in event.get("choices", [])):
            finished = True
        yield event
    if not finished:
        raise RuntimeError("chat stream ended before a finish reason or [DONE]")
