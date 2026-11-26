# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Prompts refused after the HTTP admission wait stay out of the request history."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import sys
import tempfile

from check_server import Client, wait_telemetry
from local_server import serve, server_env


def check_log(url):
    client = Client(url, None)
    model = "test model"
    oversized = {"model": model, "prompt": "0123456789 " * 6500, "max_tokens": 1}
    path = "/v1/completions"
    assert client.post(path, oversized)[0] == 400
    assert client.get("/v1/requests")[1]["requests"] == []

    blocker = {"model": model, "prompt": "0 Write integers from 1 to 2000.", "temperature": 0,
               "max_tokens": 30000, "stream": True}
    messages = {"model": model, "messages": [{"role": "user", "content": "A queued Messages request."}],
                "temperature": 0, "max_tokens": 1}
    status, prewarm = client.post("/v1/messages", dict(messages, max_tokens=0, thinking={"type": "disabled"}))
    assert status == 200, (status, prewarm)
    with ThreadPoolExecutor(max_workers=2) as pool:
        with closing(client.stream(path, blocker, stop_after=1)) as active, closing(client._conn()) as conn:
            _, first = next(active)
            assert first["choices"][0]["text"], first
            refused = pool.submit(client.post, path, oversized)
            queued = wait_telemetry(client, lambda t: t["state"]["queued"] == 1, 5)["state"]
            assert queued["queued"] == 1, queued
            conn.request("POST", "/v1/messages", json.dumps(dict(messages, stream=True)), client._headers())
            # A queued stream's HTTP headers arrive when wait_start expires. The older oversized request is
            # still waiting for the same slot and has also passed the handler's admission deadline.
            with conn.getresponse() as stream:
                assert stream.status == 200 and stream.getheader("Content-Type").startswith("text/event-stream")
                assert not refused.done()
                replacement = pool.submit(client.post, "/v1/messages", messages)
                events = [json.loads(line[6:]) for line in stream if line.startswith(b"data: ")]
                replaced_id = next(ev["message"]["id"] for ev in events if ev["type"] == "message_start")
            next(active, None)
        assert refused.result()[0] == 400
        status, reply = replacement.result()
        assert status == 200, (status, reply)

    rows = client.get("/v1/requests")[1]["requests"]
    by_id = {row["id"]: row for row in rows}
    warmed = by_id[prewarm["id"]]
    assert warmed["ttft_ms"] is None and warmed["usage"]["output_tokens"] == 0, warmed
    assert warmed["request"]["messages"] == messages["messages"], warmed
    replaced = by_id[replaced_id]
    assert replaced["status"] == "cancelled" and replaced["cancel_reason"] == "superseded", replaced
    assert replaced["request"] is None and replaced["ttft_ms"] is None, replaced
    assert by_id[reply["id"]]["request"]["messages"] == messages["messages"]
    running = next(row for row in rows if row["api"] == "completions")
    assert running["status"] == "cancelled" and running["request"]["prompt"] == blocker["prompt"], running
    assert running["ttft_ms"] is not None and running["content"] and "cancel_reason" not in running, running
    assert len(rows) == 4, [(row["api"], row["status"]) for row in rows]
    _, telemetry = client.get("/v1/telemetry")
    recent = telemetry["recent"]
    assert len(recent) == 6, recent  # includes both overlong prompts
    emitted = [r for r in recent if r["ttft_ms"] is not None]
    assert sorted((r["label"], r["status"]) for r in emitted) == [
        ("completion", "cancelled"), ("messages", by_id[reply["id"]]["status"])], recent

    # Many short (<= 1 KiB) tool arguments exceeding 256 KiB keep their full structure in /v1/requests.
    many_short = {"model": model, "prompt": "short probe", "max_tokens": 1,
                  "log_probe": [{"name": "call", "arguments": "x" * 900} for _ in range(320)]}
    st, r = client.post(path, many_short)
    assert st == 200, (st, r)
    latest = client.get("/v1/requests?limit=1")[1]["requests"][0]
    assert latest["request"] == many_short, "short tool arguments > 256 KiB should stay intact"

    # A multi-megabyte request past the hard safety ceiling still falls back to a compact summary.
    huge = {"model": model, "prompt": "huge probe", "max_tokens": 1,
            "log_probe": [{"a": "x" * 900} for _ in range(5200)]}
    st, r = client.post(path, huge)
    assert st == 200, (st, r)
    latest = client.get("/v1/requests?limit=1")[1]["requests"][0]
    assert latest["request"].get("_truncated") is True and latest["request"]["bytes"] > 4 * 1024 * 1024, latest["request"]


if __name__ == "__main__":
    exe, asset = (Path(arg).resolve() for arg in sys.argv[1:])
    with tempfile.TemporaryDirectory() as tmp:
        env = server_env(tmp, exe)
        serve(exe, tmp, env, [asset], check_log, ctx=32768,
              options=["--name", "test model", "--sessions", "1", "--no-spec"])
