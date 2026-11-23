# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Streaming tool calls emit the function name first and argument JSON incrementally on all three APIs."""
import json
import os
from pathlib import Path
import sys
import tempfile

from check_server import Client
from remembered_model import serve


WEATHER_SCHEMA = {
    "type": "object",
    "properties": {
        "location": {"type": "string", "const": "San Francisco"},
        "days": {"type": "integer", "const": 3},
    },
    "required": ["location", "days"],
    "additionalProperties": False,
}


def check_all_apis(url, label):
    client = Client(url, None)
    prompt = [{"role": "user", "content": "Weather in San Francisco for 3 days."}]
    openai_tool = {"type": "function", "function": {"name": "get_weather", "parameters": WEATHER_SCHEMA}}

    # 1. Non-streaming reference call
    st, ref = client.post("/v1/chat/completions", {
        "model": "test model", "messages": prompt, "max_tokens": 256, "temperature": 0, "frequency_penalty": 2.0,
        "tools": [openai_tool],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    })
    assert st == 200, (label, st, ref)
    ref_calls = ref["choices"][0]["message"].get("tool_calls") or []
    assert len(ref_calls) == 1 and ref_calls[0]["function"]["name"] == "get_weather", (label, ref)
    want_args = json.loads(ref_calls[0]["function"]["arguments"])
    assert want_args.get("location", "").strip() == "San Francisco" and want_args.get("days") == 3, (label, want_args, ref)

    # 2. OpenAI /v1/chat/completions streaming
    frames = list(client.stream("/v1/chat/completions", {
        "model": "test model", "messages": prompt, "max_tokens": 256, "temperature": 0, "frequency_penalty": 2.0,
        "stream": True, "tools": [openai_tool],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    }))
    tc_chunks = []
    finish = None
    for _, data in frames:
        if data == "[DONE]":
            continue
        ch = data["choices"][0]
        if ch.get("finish_reason"):
            finish = ch["finish_reason"]
        delta = ch.get("delta") or {}
        if "tool_calls" in delta:
            tc_chunks.extend(delta["tool_calls"])
    assert finish == "tool_calls", (label, finish)
    assert len(tc_chunks) >= 2, (label, "expected incremental tool_calls chunks", tc_chunks)
    first = tc_chunks[0]
    assert (first["index"] == 0 and first["id"].startswith("call_") and first["type"] == "function"
            and first["function"]["name"] == "get_weather" and first["function"]["arguments"] == ""), (label, first)
    streamed_args = "".join(c["function"]["arguments"] for c in tc_chunks[1:])
    assert json.loads(streamed_args) == want_args, (label, streamed_args, want_args)

    # 3. Anthropic /v1/messages: non-streaming reference + streaming equivalence
    anthropic_tool = {"name": "get_weather", "input_schema": WEATHER_SCHEMA}
    anthropic_body = {
        "model": "test model", "messages": prompt, "max_tokens": 256, "temperature": 0, "frequency_penalty": 2.0,
        "thinking": {"type": "disabled"},
        "tools": [anthropic_tool],
        "tool_choice": {"type": "tool", "name": "get_weather"},
    }
    st, aref = client.post("/v1/messages", anthropic_body)
    assert st == 200, (label, st, aref)
    atools = [b for b in aref["content"] if b["type"] == "tool_use"]
    assert len(atools) == 1 and atools[0]["name"] == "get_weather", (label, aref)
    want_anthropic = atools[0]["input"]
    assert want_anthropic.get("location", "").strip() == "San Francisco" and want_anthropic.get("days") == 3, (label, aref)

    events = list(client.stream("/v1/messages", dict(anthropic_body, stream=True)))
    starts = [d for ev, d in events if ev == "content_block_start" and d["content_block"]["type"] == "tool_use"]
    deltas = [d["delta"]["partial_json"] for ev, d in events
              if ev == "content_block_delta" and d["delta"]["type"] == "input_json_delta"]
    stops = [d for ev, d in events if ev == "content_block_stop"]
    mdelta = [d for ev, d in events if ev == "message_delta"]
    assert len(starts) == 1 and starts[0]["content_block"]["name"] == "get_weather", (label, starts)
    assert starts[0]["content_block"]["input"] == {}, (label, starts)
    assert len(deltas) >= 2, (label, "expected multiple input_json_delta events", deltas)
    assert json.loads("".join(deltas)) == want_anthropic, (label, deltas, want_anthropic)
    assert any(s["index"] == starts[0]["index"] for s in stops), (label, stops)
    assert mdelta and mdelta[-1]["delta"]["stop_reason"] == "tool_use", (label, mdelta)

    # 4. OpenAI /v1/responses: non-streaming reference + streaming equivalence
    resp_tool = {"type": "function", "name": "get_weather", "parameters": WEATHER_SCHEMA}
    resp_body = {
        "model": "test model", "input": "Weather in San Francisco for 3 days.", "max_output_tokens": 256,
        "temperature": 0, "frequency_penalty": 2.0,
        "tools": [resp_tool],
        "tool_choice": {"type": "function", "name": "get_weather"},
    }
    st, rref = client.post("/v1/responses", resp_body)
    assert st == 200, (label, st, rref)
    rcalls = [o for o in rref["output"] if o["type"] == "function_call"]
    assert len(rcalls) == 1 and rcalls[0]["name"] == "get_weather", (label, rref)
    want_resp = json.loads(rcalls[0]["arguments"])
    assert want_resp.get("location", "").strip() == "San Francisco" and want_resp.get("days") == 3, (label, rref)

    revents = list(client.stream("/v1/responses", dict(resp_body, stream=True)))
    added = [d["item"] for ev, d in revents if ev == "response.output_item.added" and d["item"]["type"] == "function_call"]
    rdeltas = [d["delta"] for ev, d in revents if ev == "response.function_call_arguments.delta"]
    rdone = [d for ev, d in revents if ev == "response.function_call_arguments.done"]
    idone = [d["item"] for ev, d in revents if ev == "response.output_item.done" and d["item"]["type"] == "function_call"]
    assert len(added) == 1 and added[0]["name"] == "get_weather" and added[0]["arguments"] == "", (label, added)
    assert len(rdeltas) >= 2, (label, "expected multiple function_call_arguments.delta events", rdeltas)
    joined = "".join(rdeltas)
    assert len(rdone) == 1 and rdone[0]["name"] == "get_weather" and rdone[0]["arguments"] == joined, (label, rdone, joined)
    assert len(idone) == 1 and idone[0]["status"] == "completed" and idone[0]["arguments"] == joined, (label, idone)
    assert json.loads(joined) == want_resp, (label, joined, want_resp)

    # 5. Request log records the completed tool calls cleanly across all three APIs
    st, rlog = client.get("/v1/requests")
    assert st == 200 and len(rlog["requests"]) >= 6, (label, st, rlog)
    for row in rlog["requests"][:6]:
        assert row["status"] == "tool_calls" and row["ttft_ms"] is not None, (label, row)
        calls = row.get("tool_calls") or []
        assert len(calls) == 1 and calls[0]["name"] == "get_weather", (label, row)
        assert calls[0]["arguments"].get("location", "").strip() == "San Francisco" and calls[0]["arguments"].get("days") == 3, (label, row)

    # 6. Zero-argument tool call streams "{}" cleanly
    empty_schema = {"type": "object", "properties": {}, "additionalProperties": False}
    empty_tool = {"type": "function", "function": {"name": "ping", "parameters": empty_schema}}
    eframes = list(client.stream("/v1/chat/completions", {
        "model": "test model", "messages": [{"role": "user", "content": "Ping now."}],
        "max_tokens": 256, "temperature": 0, "frequency_penalty": 2.0, "stream": True,
        "parallel_tool_calls": False,
        "tools": [empty_tool], "tool_choice": {"type": "function", "function": {"name": "ping"}},
    }))
    ecalls, efinish = [], None
    for _, data in eframes:
        if data == "[DONE]":
            continue
        ch = data["choices"][0]
        if ch.get("finish_reason"):
            efinish = ch["finish_reason"]
        delta = ch.get("delta") or {}
        if "tool_calls" in delta:
            ecalls.extend(delta["tool_calls"])
    assert efinish == "tool_calls" and ecalls and ecalls[0]["function"]["name"] == "ping", (label, efinish, ecalls)
    assert json.loads("".join(c["function"]["arguments"] for c in ecalls)) == {}, (label, ecalls)

    # 7. Truncated mid-tool-call finishes cleanly with length/max_tokens/incomplete on all three APIs
    tframes = list(client.stream("/v1/chat/completions", {
        "model": "test model", "messages": prompt, "max_tokens": 8, "temperature": 0,
        "stream": True, "tools": [openai_tool],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    }))
    tfinish = [d["choices"][0]["finish_reason"] for _, d in tframes
               if d != "[DONE]" and d["choices"][0].get("finish_reason")]
    assert tfinish == ["length"], (label, tfinish)
    tevents = list(client.stream("/v1/messages", dict(anthropic_body, max_tokens=8, stream=True)))
    tmdelta = [d["delta"]["stop_reason"] for ev, d in tevents if ev == "message_delta"]
    assert tmdelta == ["max_tokens"], (label, tmdelta)
    trevents = list(client.stream("/v1/responses", dict(resp_body, max_output_tokens=8, stream=True)))
    trstatus = [d["response"]["status"] for ev, d in trevents if ev in ("response.completed", "response.incomplete")]
    assert trstatus == ["incomplete"], (label, trstatus)


if __name__ == "__main__":
    exe, asset = (Path(arg).resolve() for arg in sys.argv[1:])
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, APPDATA=tmp, XDG_CONFIG_HOME=tmp, FLYX_BACKEND_DIR=str(exe.parent))
        template = Path(tmp) / "chat.jinja"
        # XML tool-call syntax (qwen3_coder / qwen35)
        template.write_text("{# XML tool syntax: <tool_call> <function= <parameter= #}"
                            "{% for m in messages %}{{ m.content }}|{% endfor %}", encoding="utf-8")
        serve(exe, tmp, env, [asset], lambda url: check_all_apis(url, "xml"), ctx=8192,
              options=["--name", "test model", "--no-spec", "--chat-template", template])
        # JSON tool-call syntax (hermes / qwen3)
        template.write_text("{# JSON tool syntax: <tool_call> </tool_call> #}"
                            "{% for m in messages %}{{ m.content }}|{% endfor %}", encoding="utf-8")
        serve(exe, tmp, env, [asset], lambda url: check_all_apis(url, "json"), ctx=8192,
              options=["--name", "test model", "--no-spec", "--chat-template", template])
    print("streaming_tool_calls: ok")
