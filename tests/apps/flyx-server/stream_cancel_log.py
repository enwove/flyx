# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""A stream whose client goes away mid-reply (the dashboard's Stop, a closed page) is logged with what it cost: its
prompt and output tokens and its timing, as a request that ends on its own is."""
from contextlib import closing
from pathlib import Path
import sys
import tempfile
import time

from local_server import Client, serve, server_env

MODEL = "test model"
BODIES = {
    "/v1/completions": {"model": MODEL, "prompt": "0 Write integers from 1 to 2000.", "temperature": 0,
                        "max_tokens": 30000, "stream": True},
    "/v1/chat/completions": {"model": MODEL, "messages": [{"role": "user", "content": "Count from 1 to 2000."}],
                             "temperature": 0, "max_tokens": 30000, "stream": True},
    "/v1/messages": {"model": MODEL, "messages": [{"role": "user", "content": "Count from 1 to 2000."}],
                     "thinking": {"type": "disabled"}, "temperature": 0, "max_tokens": 30000, "stream": True},
    "/v1/responses": {"model": MODEL, "input": "Count from 1 to 2000.", "reasoning": {"effort": "none"},
                      "temperature": 0, "max_output_tokens": 30000, "stream": True},
}


def logged_after(client, seen, seconds=20):
    """The first request logged that is not in `seen`, waiting up to `seconds` for it."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        for r in client.get("/v1/requests")[1]["requests"]:
            if r["id"] not in seen:
                return r
        time.sleep(0.1)
    raise AssertionError("the request was not logged")


def check(url):
    client = Client(url, None)
    for path, body in BODIES.items():
        seen = {r["id"] for r in client.get("/v1/requests")[1]["requests"]}
        received = 0
        with closing(client.stream(path, body, stop_after=8)) as events:
            for name, _ in events:
                assert name != "http_error", (path, _)
                received += 1
        assert received == 8, (path, received)
        entry = logged_after(client, seen)
        usage, timing = entry["usage"], entry.get("timing")
        assert entry["status"] == "cancelled", (path, entry["status"])
        assert usage["prompt_tokens"] > 0, (path, usage)
        # Some of the events are the stream's own (its start, a block's), not tokens: a few tokens at least.
        assert usage["output_tokens"] >= 3, (path, usage)
        assert timing and timing["decode_ms"] > 0 and timing["prefill_ms"] >= 0, (path, timing)
        print("ok", path, usage, flush=True)


if __name__ == "__main__":
    exe, asset = (Path(arg).resolve() for arg in sys.argv[1:])
    with tempfile.TemporaryDirectory() as tmp:
        env = server_env(tmp, exe)
        serve(exe, tmp, env, [asset], check, ctx=32768, options=["--name", MODEL, "--sessions", "1", "--no-spec"])
