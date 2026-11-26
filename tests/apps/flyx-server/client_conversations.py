# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Claude Code sessions and their subagents can submit identical histories concurrently.

See https://code.claude.com/docs/en/llm-gateway-protocol#request-headers
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import sys
import tempfile

from local_server import Client, serve, server_env, wait_telemetry


def post(client, path, body, headers):
    with closing(client._conn()) as conn:
        conn.request("POST", path, json.dumps(body), {**client._headers(), **headers})
        with conn.getresponse() as reply:
            data = json.load(reply)
            if path == "/v1/messages" and reply.status == 409:
                assert reply.getheader("x-should-retry") == "false" and data["error"]["type"] == "conflict_error"
            return reply.status, data


def queued_pair(client, pool, blocker, first, second):
    with closing(client.stream("/v1/completions", blocker, stop_after=1)) as stream:
        assert next(stream)[1]["choices"][0]["text"]
        a = pool.submit(first)
        state = wait_telemetry(client, lambda t: a.done() or t["state"]["queued"] == 1, 5)["state"]
        assert state["queued"] == 1, state
        b = pool.submit(second)
        state = wait_telemetry(client, lambda t: a.done() or t["state"]["queued"] == 2, 5)["state"]
        assert a.done() or state["queued"] == 2, state
        next(stream, None)  # release the slot after both requests have been submitted
    return a.result(), b.result(), state["queued"]


def running_pair(client, pool, label, first, second):
    a = pool.submit(first)
    state = wait_telemetry(client, lambda t: a.done() or t["state"].get("generated", 0) >= 4, 5)["state"]
    assert state.get("label") == label and state["generated"] >= 4, state
    b = pool.submit(second)
    state = wait_telemetry(client, lambda t: a.done() or t["state"]["queued"] == 1, 5)["state"]
    return a.result(), b.result(), state["queued"]


def check(url):
    client = Client(url, None)
    model = "test model"
    long_prompt = "0 Write integers from 1 to 2000."
    blocker = {"model": model, "prompt": long_prompt, "stream": True, "temperature": 0, "max_tokens": 8000}

    def message(identity, content, tokens=8):
        session, agent = identity
        headers = {}
        if session:
            headers["x-claude-code-session-id"] = session
        if agent:
            headers["x-claude-code-agent-id"] = agent
        body = {"model": model, "max_tokens": tokens, "temperature": 0, "messages": [{"role": "user", "content": content}]}
        return post(client, "/v1/messages", body, headers)

    main = ("one", ""), ("two", "")
    agent = ("one", "worker"), ("one", "reviewer")
    unknown = ("", "")
    status, serial = message(unknown, "Hello")
    assert status == 200 and serial["content"], serial
    failures = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for first, second, text, expected in (
            (*main, "Hello", 200), (*agent, "Hello", 200), (main[0], agent[0], "Hello", 200),
            (("one:two", "three"), ("one", "two:three"), "Hello", 200),
            (agent[0], agent[0], "Hello", 409), (main[0], unknown, "Hello", 409),
            (unknown, main[0], "Hello", 409), (unknown, unknown, "Hello", 409),
            (("", "worker"), agent[1], "Hello", 409),
            (main[0], main[0], "A different request", 200),
        ):
            (first_status, first_reply), (second_status, second_reply), queued = queued_pair(
                client, pool, blocker, lambda: message(first, "Hello"), lambda: message(second, text))
            if (first_status, second_status) != (expected, 200):
                failures.append(("queued", first, second, first_status, second_status))
            elif expected == 200:
                assert queued == 2, queued
                if text == "Hello":
                    assert first_reply["content"] == second_reply["content"] == serial["content"]
                    for reply in (first_reply, second_reply):
                        assert reply["usage"]["output_tokens"] == serial["usage"]["output_tokens"]

        for first, second, expected in ((*main, 200), (*agent, 200), (agent[0], agent[0], 409)):
            (first_status, first_reply), (second_status, _), queued = running_pair(
                client, pool, "messages", lambda: message(first, long_prompt, 1024), lambda: message(second, long_prompt))
            if (first_status, second_status) != (expected, 200):
                failures.append(("running", first, second, first_status, second_status))
            elif expected == 200:
                assert queued == 1 and first_reply["usage"]["output_tokens"] == 1024, queued

        # A live Messages stream superseded by a replacement for the same conversation ends cleanly
        # with message_stop (no SSE error event, which would trigger SDK retries).
        with closing(client._conn()) as conn:
            body = {"model": model, "max_tokens": 1024, "temperature": 0, "stream": True,
                    "messages": [{"role": "user", "content": long_prompt}]}
            conn.request("POST", "/v1/messages", json.dumps(body),
                         {**client._headers(), "x-claude-code-session-id": "one", "x-claude-code-agent-id": "worker"})
            wait_telemetry(client, lambda t: t["state"].get("generated", 0) >= 4, 5)
            rep_st, rep_body = message(agent[0], long_prompt, tokens=1)
            assert rep_st == 200 and rep_body["usage"]["output_tokens"] == 1, (rep_st, rep_body)
            with conn.getresponse() as old_stream:
                events = [json.loads(line[6:]) for line in old_stream if line.startswith(b"data: ")]
                kinds = [ev["type"] for ev in events]
                assert old_stream.status == 200 and "content_block_delta" in kinds and kinds[-1:] == ["message_stop"] \
                    and "error" not in kinds, kinds
    assert not failures, failures


def serve_cases(check):
    exe, asset = (Path(arg).resolve() for arg in sys.argv[1:])
    with tempfile.TemporaryDirectory() as tmp:
        env = server_env(tmp, exe)
        template = Path(tmp) / "chat.jinja"
        template.write_text("{% for m in messages %}{{ m.content }}{% endfor %}", encoding="utf-8")
        serve(exe, tmp, env, [asset], check, ctx=8192,
              options=["--name", "test model", "--sessions", "1", "--no-spec", "--chat-template", template])


if __name__ == "__main__":
    serve_cases(check)
