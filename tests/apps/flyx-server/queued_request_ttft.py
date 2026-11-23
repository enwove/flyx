# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The scheduler emits the first output after the queue wait, before the HTTP handler observes it."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import os
from pathlib import Path
import sys
import tempfile

from check_server import Client, wait_telemetry
from remembered_model import serve


def check_times(url):
    client = Client(url, None)
    routes = {
        "messages": {"messages": [{"role": "user", "content": "Oak title."}], "max_tokens": 1},
        "chat/completions": {"messages": [{"role": "user", "content": "Birch title."}], "max_tokens": 1},
        "completions": {"prompt": "Pine title.", "max_tokens": 1},
        "responses": {"input": "Maple title.", "max_output_tokens": 1},
    }

    def post(route, body):
        status, reply = client.post("/v1/" + route, dict(body, model="test model", temperature=0))
        assert status == 200, (route, status, reply)
        return reply["id"]

    long_request = {"model": "test model", "prompt": "0 Write integers from 1 to 2000.", "temperature": 0,
                    "max_tokens": 8000, "stream": True}
    with ThreadPoolExecutor(max_workers=len(routes)) as pool:
        with closing(client.stream("/v1/completions", long_request, stop_after=1)) as stream:
            _, chunk = next(stream)
            assert chunk["choices"][0]["text"], chunk
            pending = {route: pool.submit(post, route, body) for route, body in routes.items()}

            def wait_queued(min_generated):
                def queued(t):
                    state = t["state"]
                    return (state["running"] == 1 and state["queued"] == len(routes)
                            and state["generated"] >= min_generated)

                snapshot = wait_telemetry(client, lambda t: queued(t) or any(job.done() for job in pending.values()), 30)
                assert queued(snapshot), snapshot["state"]
                return snapshot["state"]["generated"]

            # Keep the slot busy for more decode steps after every HTTP handler is waiting.
            when_queued = wait_queued(1)
            wait_queued(when_queued + 64)
            next(stream, None)  # stop_after closes the connection and releases the slot
        ids = {route: job.result() for route, job in pending.items()}

    status, log = client.get("/v1/requests")
    assert status == 200, (status, log)
    rows = {row["id"]: row for row in log["requests"]}
    _, telemetry = client.get("/v1/telemetry")
    labels = {"messages": "messages", "completions": "completion", "chat/completions": "chat", "responses": "responses"}
    failures = []
    for route, id_ in ids.items():
        row = rows[id_]
        queue_ms, ttft_ms = row["timing"]["queue_ms"], row["ttft_ms"]
        recent = next(r for r in telemetry["recent"] if r["label"] == labels[route] and r["status"] == row["status"]
                      and r["queue_ms"] == queue_ms)
        emitted_ms = recent["ttft_ms"]
        print(f"{route}: queue_ms={queue_ms:.3f}, emitted_ms={emitted_ms}, handler_ms={ttft_ms}")
        if emitted_ms is None or ttft_ms is None or not 0 < queue_ms <= emitted_ms <= ttft_ms:
            failures.append(route)
    assert not failures, f"TTFT omitted queue time for {', '.join(failures)}"


def without_request_log(url):
    client = Client(url, None)
    messages = {"model": "test model", "messages": [{"role": "user", "content": "Oak title."}],
                "thinking": {"type": "disabled"}}
    for budget in (0, 1):
        status, reply = client.post("/v1/messages", dict(messages, max_tokens=budget, temperature=0))
        assert status == 200, reply
    assert client.get("/v1/requests")[1] == {"enabled": False, "requests": []}
    latest, warm = client.get("/v1/telemetry?recent=2")[1]["recent"]
    assert latest["ttft_ms"] >= latest["queue_ms"] and warm["ttft_ms"] is None, (latest, warm)
    tool = {"name": "ping", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}
    status, reply = client.post("/v1/chat/completions", {
        "model": "test model", "messages": messages["messages"], "max_tokens": 128, "temperature": 0,
        "tools": [{"type": "function", "function": tool}], "tool_choice": {"type": "function", "function": {"name": "ping"}}})
    assert status == 200, reply
    msg = reply["choices"][0]["message"]
    assert not msg.get("reasoning_content") and not msg.get("content") and msg["tool_calls"], msg
    latest = client.get("/v1/telemetry?recent=1")[1]["recent"][0]
    assert latest["tool_calls"] == 1 and latest["ttft_ms"] >= latest["queue_ms"], latest


def reasoning_only(url):
    client = Client(url, None)
    status, reply = client.post("/v1/chat/completions", {
        "model": "test model", "temperature": 0, "max_tokens": 1, "messages": [{"role": "user", "content": "Ping."}]})
    assert status == 200, reply
    msg = reply["choices"][0]["message"]
    assert msg.get("reasoning_content") and not msg.get("content") and not msg.get("tool_calls"), msg
    rec = client.get("/v1/telemetry?recent=1")[1]["recent"][0]
    assert rec["ttft_ms"] >= rec["queue_ms"], rec


if __name__ == "__main__":
    exe, asset = (Path(arg).resolve() for arg in sys.argv[1:])
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, APPDATA=tmp, XDG_CONFIG_HOME=tmp, FLYX_BACKEND_DIR=str(exe.parent))
        # Distinct rendered conversations let all four requests wait without replacing each other.
        template = Path(tmp) / "chat.jinja"
        template.write_text("{% for m in messages %}{{ m.role }}:{{ m.content }}|{% endfor %}", encoding="utf-8")
        serve(exe, tmp, env, [asset], check_times, ctx=8192,
              options=["--name", "test model", "--sessions", "1", "--no-spec", "--chat-template", template])
        template.write_text("{# XML tool syntax: <tool_call> <function= <parameter= #}"
                            "{% for m in messages %}{{ m.content }}|{% endfor %}", encoding="utf-8")
        serve(exe, tmp, env, [asset], without_request_log, ctx=8192,
              options=["--name", "test model", "--no-spec", "--no-request-log", "--chat-template", template])
        qwen = Path(__file__).resolve().parents[2] / "core/render/templates/qwen3.5.jinja"
        serve(exe, tmp, env, [asset], reasoning_only, options=["--name", "test model", "--no-spec", "--chat-template", qwen])
