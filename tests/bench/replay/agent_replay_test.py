# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""A streamed tool call split across chunks and a server that takes time to send headers."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "bench" / "replay"))
import agent_replay
from http_api import chat_events


def chunks():
    yield {"role": "assistant"}
    yield {"tool_calls": [{"index": 0, "id": "a", "type": "function",
                           "function": {"name": "read_file", "arguments": '{"path":'}}]}
    yield {"tool_calls": [{"index": 1, "id": "b", "type": "function",
                           "function": {"name": "search", "arguments": "{}"}}]}
    yield {"tool_calls": [{"index": 0, "function": {"arguments": '"main.py"}'}}]}


class Response:
    status = 200

    def __init__(self, now):
        self.now = now

    def __iter__(self):
        for at, delta in zip((12, 12.25, 12.3, 12.4), chunks()):
            self.now[0] = at
            yield ("data: " + json.dumps({"choices": [{"delta": delta}]}) + "\n").encode()
        self.now[0] = 12.5
        yield b"data: [DONE]\n"


if __name__ == "__main__":
    # Send at 10, receive headers at 12, first tool delta at 12.25, stream ends at 12.5.
    now = [10.0]

    def headers(request, **kwargs):
        assert request.full_url == "https://localhost/service/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer local-test"
        now[0] = 12.0
        return nullcontext(Response(now))

    with patch.dict("os.environ", {"FLYX_KEY": "local-test"}), \
            patch.object(agent_replay.urllib.request, "urlopen", headers), \
            patch.object(agent_replay.time, "perf_counter", lambda: now[0]):
        _, _, tools, _, first, total = agent_replay.chat_stream("https://localhost/service/", {})
    assert (first, total) == (2.25, 2.5), (first, total)
    assert tools == [{"id": "a", "function": {"name": "read_file", "arguments": '{"path":"main.py"}'}},
                     {"id": "b", "function": {"name": "search", "arguments": "{}"}}], tools
    partial = b'data: {"choices":[{"delta":{"content":"hi"}}]}'
    try:
        list(chat_events([partial]))
    except RuntimeError as exc:
        assert "finish reason" in str(exc)
    else:
        raise AssertionError("an abruptly closed stream was reported as complete")
    finish = b'data: {"choices":[{"finish_reason":"stop","delta":{}}]}'
    assert len(list(chat_events([partial, finish]))) == 2
