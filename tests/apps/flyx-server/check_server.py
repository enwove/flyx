# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""End-to-end checks of flyx-server's OpenAI (chat completions, Responses) and Anthropic APIs (standard library only).

    flyx-server MODEL.flyx --port 8080 &
    python tests/apps/flyx-server/check_server.py --url http://127.0.0.1:8080

Needs a real model (tool calls must be meaningful); every check prints ok / FAIL.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import tempfile
import threading
import time
import urllib.parse

FAILS = 0
# Room for a reply with thinking on (the question of the streamed checks); one that still runs out is checked as cut.
THINK_TOKENS = 4096


def check(ok: bool, what: str, detail: str = ""):
    global FAILS
    print(f"  {'ok  ' if ok else 'FAIL'} {what}" + (f"  [{detail}]" if detail and not ok else ""))
    FAILS += 0 if ok else 1


class Client:
    def __init__(self, url: str, key: str | None):
        u = urllib.parse.urlparse(url)
        self.host, self.port = u.hostname, u.port or 80
        self.key = key

    def _conn(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=600)

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = "Bearer " + self.key
        return h

    def get_raw(self, path, headers=None):
        c = self._conn()
        try:
            c.request("GET", path, headers={**self._headers(), **(headers or {})})
            r = c.getresponse()
            return r.status, r.headers, r.read()
        finally:
            c.close()

    def get(self, path):
        status, _, data = self.get_raw(path)
        return status, json.loads(data or b"null")

    def post(self, path, body, headers=None):
        c = self._conn()
        c.request("POST", path, body=json.dumps(body), headers={**self._headers(), **(headers or {})})
        r = c.getresponse()
        return r.status, json.loads(r.read() or b"null")

    def delete(self, path):
        c = self._conn()
        c.request("DELETE", path, headers=self._headers())
        r = c.getresponse()
        return r.status, json.loads(r.read() or b"null")

    def stream(self, path, body, stop_after=None):
        """Yields (event_name, data) of an SSE response; stop_after closes the connection early."""
        c = self._conn()
        c.request("POST", path, body=json.dumps(body), headers=self._headers())
        r = c.getresponse()
        if r.status != 200:
            yield "http_error", {"status": r.status, "body": r.read().decode()}
            return
        event, n = None, 0
        while True:
            line = r.readline()
            if not line:
                break
            line = line.decode().rstrip("\r\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
                yield event, (data if data == "[DONE]" else json.loads(data))
                event = None
                n += 1
                if stop_after and n >= stop_after:
                    c.close()
                    return


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
TIME = {"type": "function", "function": {
    "name": "get_time", "description": "Current time in a timezone.",
    "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]}}}


def fits_weather(args: dict) -> bool:
    return (isinstance(args.get("city"), str) and set(args) <= {"city", "unit"} and
            args.get("unit", "celsius") in ("celsius", "fahrenheit"))


def tool_choice_checks(cl: Client, model: str):
    """tool_choice and response_format hold the reply to a grammar (core/grammar)."""
    print("tool_choice, response_format")
    def ask(text, **kw):
        st, r = cl.post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": text}],
                                                 "max_tokens": 1024, "temperature": 0, **kw})
        if st != 200:
            return st, r, None, []
        m = r["choices"][0]["message"]
        return st, r, m, [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                          for c in m.get("tool_calls") or []]

    st, r, m, calls = ask("Hello there!", tools=[WEATHER, TIME], tool_choice="required")
    check(st == 200 and len(calls) >= 1 and r["choices"][0]["finish_reason"] == "tool_calls",
          "tool_choice required: a call even for a greeting", json.dumps(r)[:300])
    st, r, m, calls = ask("What time is it in Tokyo?", tools=[WEATHER, TIME],
                          tool_choice={"type": "function", "function": {"name": "get_weather"}})
    check(st == 200 and len(calls) >= 1 and all(n == "get_weather" and fits_weather(a) for n, a in calls),
          "tool_choice forced: the named tool, arguments fit its schema", json.dumps(calls))
    st, r, m, calls = ask("Weather in Rome and in Oslo, in celsius?", tools=[WEATHER], tool_choice="required",
                          parallel_tool_calls=False)
    check(st == 200 and len(calls) == 1 and fits_weather(calls[0][1]), "parallel_tool_calls false: one call",
          json.dumps(calls))
    st, r, m, calls = ask("What's the weather in Paris? Use the tool.", tools=[WEATHER], tool_choice="none")
    check(st == 200 and not calls, "tool_choice none: no call", json.dumps(r)[:300])
    st, r, m, calls = ask("Give me a person with a name and an age.", chat_template_kwargs={"enable_thinking": False},
                          response_format={"type": "json_schema", "json_schema": {"name": "person", "schema": {
                              "type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                              "required": ["name", "age"], "additionalProperties": False}}})
    try:
        person = json.loads(m.get("content") or "") if m else None
    except ValueError:
        person = None
    check(isinstance(person, dict) and set(person) == {"name", "age"} and isinstance(person["age"], int),
          "response_format json_schema: the answer fits the schema", (m or {}).get("content", "")[:200])
    st, r, m, calls = ask("Hi", tools=[WEATHER], tool_choice={"type": "function", "function": {"name": "nope"}})
    check(st == 400 and "error" in r, "tool_choice naming an unknown tool -> 400", json.dumps(r)[:200])


def streaming_flag_checks(cl: Client, model: str):
    chat = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}
    cases = (("chat/completions", chat), ("completions", {"prompt": "Hi", "max_tokens": 1}),
             ("responses", {"input": "Hi", "max_output_tokens": 1}), ("messages", chat))
    for api, body in cases:
        path = "/v1/" + api
        st, result = cl.post(path, dict(body, model=model, stream=None))
        check(st == 200 and result.get("usage") is not None, api + ": null stream uses the nonstream default")
        st, result = cl.post(path, dict(body, model=model, stream="false"))
        check(st == 400 and result["error"]["type"] == "invalid_request_error", api + ": a string stream flag is invalid")
    for invalid in ("yes", {"include_usage": "yes"}):
        st, result = cl.post("/v1/chat/completions", dict(chat, model=model, stream_options=invalid))
        check(st == 400 and result["error"]["type"] == "invalid_request_error", "invalid stream_options")


def openai_checks(cl: Client, model: str):
    print("OpenAI /v1/chat/completions")
    st, r = cl.post("/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}],
        "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    msg = r["choices"][0]["message"] if st == 200 else {}
    check(st == 200 and "391" in (msg.get("content") or ""), "greedy answer, thinking off", json.dumps(r)[:300])
    check(st == 200 and r["choices"][0]["finish_reason"] == "stop", "finish_reason stop")
    check(st == 200 and r["usage"]["completion_tokens"] > 0, "usage reported")

    # Streaming with thinking.  The protocol is checked whichever way the reply ends: a model may still be thinking when
    # the tokens run out (the NVFP4 Qwen3.5-4B, greedy, went back and forth over its wording past 1024: issue #4), and
    # then the stream ends with finish_reason "length" and no answer; the answer is checked where there is one.
    reasoning, content, finish, usage = "", "", None, None
    for ev, d in cl.stream("/v1/chat/completions", {
            "model": model, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": THINK_TOKENS,
            "temperature": 0, "messages": [{"role": "user", "content": "Is 91 a prime number? Answer yes or no."}]}):
        if d == "[DONE]":
            break
        if ev == "http_error":
            check(False, "stream", str(d))
            return
        if d.get("usage"):
            usage = d["usage"]
        for ch in d.get("choices", []):
            reasoning += ch["delta"].get("reasoning_content") or ""
            content += ch["delta"].get("content") or ""
            finish = ch.get("finish_reason") or finish
    check(len(reasoning) > 20 and "</think>" not in reasoning, "stream: thinking in reasoning_content", reasoning[-200:])
    if finish == "length":
        print(f"  note stream: the reply ran out of its {THINK_TOKENS} tokens while thinking; no answer to check")
        check("<think>" not in content and (usage or {}).get("completion_tokens") == THINK_TOKENS,
              "stream: a reply cut at its limit, and the usage chunk", json.dumps(usage))
    else:
        check("no" in content.lower() and "<think>" not in content, "stream: answer in content", content[:200])
        check(finish == "stop" and usage is not None, "stream: finish and usage chunk")

    # tool call, then the tool result
    msgs = [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}]
    st, r = cl.post("/v1/chat/completions", {"model": model, "messages": msgs, "tools": [WEATHER],
                                             "max_tokens": 1024, "temperature": 0})
    calls = r["choices"][0]["message"].get("tool_calls") if st == 200 else None
    ok = bool(calls) and calls[0]["function"]["name"] == "get_weather"
    args = json.loads(calls[0]["function"]["arguments"]) if ok else {}
    check(ok and "paris" in str(args.get("city", "")).lower(), "tool call with typed arguments", json.dumps(r)[:400])
    check(st == 200 and r["choices"][0]["finish_reason"] == "tool_calls", "finish_reason tool_calls")
    if ok:
        m = r["choices"][0]["message"]
        msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls,
                     "reasoning_content": m.get("reasoning_content")})
        msgs.append({"role": "tool", "tool_call_id": calls[0]["id"], "content": '{"temp_c": 18, "sky": "rain"}'})
        st, r2 = cl.post("/v1/chat/completions", {"model": model, "messages": msgs, "tools": [WEATHER],
                                                  "max_tokens": 1024, "temperature": 0})
        text = r2["choices"][0]["message"].get("content") or "" if st == 200 else ""
        check("18" in text, "answer uses the tool result", text[:200])
        cached = r2["usage"]["prompt_tokens_details"]["cached_tokens"] if st == 200 else 0
        prompt = r2["usage"]["prompt_tokens"] if st == 200 else 1
        check(cached >= prompt - 64, f"follow-up reuses the session ({cached} of {prompt} prompt tokens cached)")

    # stop strings
    st, r = cl.post("/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": "Count from 1 to 10, comma separated."}],
        "max_tokens": 64, "temperature": 0, "stop": ["5"], "chat_template_kwargs": {"enable_thinking": False}})
    c = r["choices"][0]["message"]["content"] if st == 200 else ""
    check(st == 200 and "4" in c and "5" not in c and r["choices"][0]["finish_reason"] == "stop",
          "stop string ends the reply before it", repr(c))

    # sampling: same seed, same text; different seed, usually different
    def sample(seed):
        st, r = cl.post("/v1/chat/completions", {
            "model": model, "messages": [{"role": "user", "content": "Write a four-line poem about rivers."}],
            "max_tokens": 48, "temperature": 1.0, "top_p": 0.95, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False}})
        return r["choices"][0]["message"]["content"] if st == 200 else None
    a, b, c = sample(7), sample(7), sample(8)
    check(a is not None and a == b, "same seed gives the same sample")
    check(a != c, "another seed gives another sample")

    # Penalties. A reply that would repeat itself changes under a frequency
    # penalty; requests without thinking get Qwen3.5's presence penalty of 1.5 unless they say
    # otherwise (the same seed: as with 1.5 spelled out, not as with 0).
    def with_penalty(prompt, **kw):
        st, r = cl.post("/v1/chat/completions", {
            "model": model, "max_tokens": 96, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}], **kw})
        return r["choices"][0]["message"]["content"] if st == 200 else None
    echo = "Write the word echo forty times, separated by spaces."
    plain = with_penalty(echo, temperature=0, presence_penalty=0)
    pushed = with_penalty(echo, temperature=0, frequency_penalty=2)
    check(plain is not None and plain.lower().count("echo") >= 20 and pushed is not None and
          pushed.lower().count("echo") < plain.lower().count("echo"),
          "frequency_penalty: fewer repeats of a repeated word",
          f"{plain and plain.lower().count('echo')} then {pushed and pushed.lower().count('echo')}")
    # Give the model a choice of words: even a penalized token can win every time if the prompt requires it.
    names = "List twenty possible names for a pet cat, one per line."
    by_default = with_penalty(names, seed=7)
    spelled = with_penalty(names, seed=7, presence_penalty=1.5)
    without = with_penalty(names, seed=7, presence_penalty=0)
    check(by_default is not None and by_default == spelled and by_default != without,
          "without thinking the default presence penalty is 1.5", f"{by_default!r} / {spelled!r} / {without!r}")
    st, r = cl.post("/v1/chat/completions", {"model": model, "max_tokens": 4, "presence_penalty": 3,
                                             "messages": [{"role": "user", "content": "hi"}]})
    check(st == 400 and "error" in r, "presence_penalty out of range -> 400", json.dumps(r)[:200])
    # flyx's relaxed acceptance of drafts (lossy, opt-in): taken, and refused out of range
    st, r = cl.post("/v1/chat/completions", {"model": model, "max_tokens": 4, "accept_min_p": 0.2,
                                             "messages": [{"role": "user", "content": "Hi"}]})
    check(st == 200 and r.get("choices"), "accept_min_p 0.2 -> 200", json.dumps(r)[:200])
    st, r = cl.post("/v1/chat/completions", {"model": model, "max_tokens": 4, "accept_typical": 2,
                                             "messages": [{"role": "user", "content": "Hi"}]})
    check(st == 400 and "error" in r, "accept_typical out of range -> 400", json.dumps(r)[:200])

    print("OpenAI /v1/completions")
    raw = {"model": model, "prompt": "The first twenty prime numbers are", "temperature": 0}
    st, default = cl.post("/v1/completions", raw)
    st_null, explicit_null = cl.post("/v1/completions", dict(raw, max_tokens=None))
    check(st == st_null == 200 and default["choices"] == explicit_null["choices"] and
          default["usage"]["completion_tokens"] == explicit_null["usage"]["completion_tokens"] <= 16,
          "omitted and null max_tokens use the 16-token raw completion default")
    st, r = cl.post("/v1/completions", {"model": model, "prompt": "The capital of France is", "max_tokens": 8,
                                        "temperature": 0})
    check(st == 200 and "Paris" in r["choices"][0]["text"], "raw completion", json.dumps(r)[:200])
    events = list(cl.stream("/v1/completions", {"model": model, "prompt": "The capital of France is",
                                                "max_tokens": 8, "temperature": 0, "stream": True,
                                                "stream_options": {"include_usage": True}}))
    chunks = [d for _, d in events if isinstance(d, dict) and d.get("choices")]
    text = "".join(c["choices"][0]["text"] for c in chunks)
    usage = [d for _, d in events if isinstance(d, dict) and d.get("usage")]
    check(text == r["choices"][0]["text"] and chunks[-1]["choices"][0]["finish_reason"] is not None and
          len(usage) == 1 and events[-1][1] == "[DONE]",
          "raw completion streamed: the same text, a finish reason, usage, [DONE]", json.dumps(events)[:300])

    print("cancel")
    t0 = time.time()
    n = sum(1 for _ in cl.stream("/v1/chat/completions", {
        "model": model, "stream": True, "max_tokens": 2000, "temperature": 0,
        "messages": [{"role": "user", "content": "Write a very long story about a lighthouse keeper."}]},
        stop_after=5))
    st, r = cl.post("/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": False}})
    check(st == 200 and time.time() - t0 < 60, f"a closed stream is cancelled ({time.time() - t0:.1f} s)")

    st, r = cl.post("/v1/chat/completions", {"model": model, "messages": []})
    check(st == 400 and "error" in r, "bad request -> 400 with an error object")


def response_input_pages(cl: Client, model: str):
    source = [{"role": role, "content": text} for role, text in
              (("user", "Hello"), ("assistant", "Hi"), ("user", "Bye"))]
    source[-1] = {"type": "message", "id": "msg_last", "role": "user", "content": [{"type": "input_text", "text": "Bye"}]}
    parts = [[{"type": "input_text", "text": "Hello"}], [{"type": "output_text", "text": "Hi", "annotations": []}],
             [{"type": "input_text", "text": "Bye"}]]
    st, r = cl.post("/v1/responses", {"model": model, "input": source, "max_output_tokens": 1})
    check(st == 200, "create a response with three input items")
    if st != 200:
        return
    path = "/v1/responses/" + r["id"] + "/input_items"
    for order in ("asc", "desc"):
        url = path + "?order=" + order + "&limit=2"
        expected = parts if order == "asc" else parts[::-1]
        st, first = cl.get(url)
        check(st == 200 and first["has_more"] and len(first["data"]) == 2, order + ": first input page")
        st, last = cl.get(url + "&after=" + first["last_id"])
        items = first["data"] + last["data"]
        check(st == 200 and all(i.get("type") == "message" and i.get("status") == "completed" for i in items)
              and [i["content"] for i in items] == expected
              and not last["has_more"] and last["first_id"] == last["last_id"] == items[-1]["id"],
              order + ": after continues past the cursor and finishes")
        st, empty = cl.get(url + "&after=" + last["last_id"])
        check(st == 200 and empty["data"] == [] and not empty["has_more"]
              and empty["first_id"] is None and empty["last_id"] is None, order + ": after the last item")
    st, _ = cl.get(path + "?after=unknown_item")
    check(st == 400, "a cursor outside this response returns an error")
    st, page = cl.get(path + "?limit=100")
    check(st == 200 and [i["content"] for i in page["data"]] == parts[::-1] and page["first_id"] == "msg_last"
          and not page["has_more"], "the maximum page size and the default descending order")
    for params in ("limit=0", "limit=101", "limit=2x", "limit=99999999999999999999999", "order=unknown"):
        st, error = cl.get(path + "?" + params)
        check(st == 400 and error["error"]["type"] == "invalid_request_error", "invalid input page: " + params)


def response_item_ids(cl: Client, model: str):
    source = [
        {"role": "user", "id": "", "content": "Look up a label."},
        {"type": "function_call", "id": "item_shared", "call_id": "call_lookup", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "id": "item_shared", "call_id": "call_lookup", "output": "BLUE"},
        {"type": "reasoning", "id": None, "summary": [{"type": "summary_text", "text": "Read the label."}]},
        {"role": "assistant", "id": "msg_reserved_later", "content": "BLUE"},
        {"role": "user", "id": 42, "content": "Repeat it."},
    ]
    st, r = cl.post("/v1/responses", {"model": model, "input": source, "max_output_tokens": 1})
    check(st == 200, "replayed item ids are accepted")
    if st != 200:
        return
    path = "/v1/responses/" + r["id"] + "/input_items"
    _, full = cl.get(path + "?order=asc")
    rows = full["data"]
    ids = [i["id"] for i in rows]
    check(len(set(ids)) == len(source) and ids[1] == "item_shared" and ids[4] == "msg_reserved_later"
          and all(ids[i].startswith(prefix) for i, prefix in ((0, "msg_"), (2, "fco_"), (3, "rs_"), (5, "msg_")))
          and rows[1]["call_id"] == rows[2]["call_id"] == "call_lookup",
          "stored input items have unique typed ids and keep the tool result linked to its call")
    for order in ("asc", "desc"):
        after, seen = "", []
        for _ in source:
            st, page = cl.get(path + "?order=" + order + "&limit=1" + after)
            if st != 200 or not page["data"]:
                break
            cursor = page["data"][-1]["id"]
            seen.append(cursor)
            if not page["has_more"]:
                break
            after = "&after=" + urllib.parse.quote(cursor, safe="")
        check(st == 200 and not page["has_more"] and seen == (ids if order == "asc" else ids[::-1]),
              order + ": replayed item ids page through the whole input")


def response_instruction_checks(cl: Client, model: str):
    """Top-level instructions apply to one request. An input developer message stays in the chain."""
    first = [{"role": "developer", "content": "Use short sentences."}, {"role": "user", "content": "Say hello."}]
    options = {"model": model, "max_output_tokens": 3, "reasoning": {"effort": "none"}, "temperature": 0}
    st, initial = cl.post("/v1/responses", dict(options, input=first, instructions="Answer in English."))
    check(st == 200, "current instructions and an input developer message work together", str(initial)[:300])
    if st != 200:
        return
    st, listed = cl.get("/v1/responses/" + initial["id"] + "/input_items?order=asc")
    check(st == 200 and [i.get("role") for i in listed["data"]] == ["developer", "user"],
          "input_items preserves the original roles and excludes top-level instructions")
    opts = dict(options, instructions="Answer in French.", store=False)
    st, follow = cl.post("/v1/responses", dict(opts, previous_response_id=initial["id"], input="Again."))
    full = first + initial["output"] + [{"role": "user", "content": "Again."}]
    stream = list(cl.stream("/v1/responses", dict(opts, input=full, stream=True)))
    end = stream[-1][1].get("response", {}) if stream else {}

    def parts(r):
        return [(i["type"], i.get("content")) for i in r.get("output", [])]

    check(st == 200 and end.get("instructions") == follow.get("instructions") == opts["instructions"] and
          parts(follow) == parts(end) and follow["usage"]["input_tokens"] == end["usage"]["input_tokens"],
          "stored continuation swaps instructions and matches streaming the original items")


def responses_checks(cl: Client, model: str):
    """The OpenAI Responses API: items, the event stream, previous_response_id, the store."""
    print("OpenAI /v1/responses")
    response_input_pages(cl, model)
    response_item_ids(cl, model)
    response_instruction_checks(cl, model)
    fn = dict(WEATHER["function"], type="function")  # the flat Responses form

    def texts(r):
        return "".join(c["text"] for o in r.get("output", []) if o["type"] == "message" for c in o["content"])

    st, r = cl.post("/v1/responses", {"model": model, "input": "What is 17 * 23? Answer with the number only.",
                                      "reasoning": {"effort": "none"}, "max_output_tokens": 64, "temperature": 0})
    check(st == 200 and r["object"] == "response" and r["status"] == "completed" and r.get("completed_at") and
          "391" in texts(r) and
          not any(o["type"] == "reasoning" for o in r["output"]), "an answer, reasoning effort none", json.dumps(r)[:300])
    check(st == 200 and r["usage"]["output_tokens"] > 0 and r["usage"]["output_tokens_details"]["reasoning_tokens"] == 0,
          "usage reported")

    # A call, its output by previous_response_id, and the same turn sent back in full.
    question = "What's the weather in Paris right now? Use the tool."
    st, r = cl.post("/v1/responses", {"model": model, "input": question, "tools": [fn], "temperature": 0,
                                      "max_output_tokens": 1024})
    out = r.get("output", []) if st == 200 else []
    calls = [o for o in out if o["type"] == "function_call"]
    args = json.loads(calls[0]["arguments"]) if calls else {}
    check(bool(calls) and calls[0]["name"] == "get_weather" and "paris" in str(args.get("city", "")).lower(),
          "function_call item with arguments", json.dumps(r)[:400])
    check(any(o["type"] == "reasoning" and o["content"][0]["text"] for o in out) and
          r["usage"]["output_tokens_details"]["reasoning_tokens"] > 0, "reasoning item and reasoning_tokens")
    if calls:
        result = {"type": "function_call_output", "call_id": calls[0]["call_id"],
                  "output": "{\"temp_c\": 18, \"sky\": \"rain\"}"}
        st, r2 = cl.post("/v1/responses", {"model": model, "previous_response_id": r["id"], "input": [result],
                                           "tools": [fn], "temperature": 0, "max_output_tokens": 1024})
        check(st == 200 and "18" in texts(r2), "previous_response_id + function_call_output: the answer uses it",
              json.dumps(r2)[:300])
        prompt = r2["usage"]["input_tokens"] if st == 200 else -1
        cached = r2["usage"]["input_tokens_details"]["cached_tokens"] if st == 200 else 0
        check(cached >= prompt - 64, f"the continuation reuses the session ({cached} of {prompt} input tokens cached)")
        st, r3 = cl.post("/v1/responses", {"model": model, "store": False, "tools": [fn], "temperature": 0,
                                           "max_output_tokens": 16,
                                           "input": [{"role": "user", "content": question}] + out + [result]})
        check(st == 200 and r3["usage"]["input_tokens"] == prompt,
              "the items sent back in full make the same prompt as previous_response_id")
        rid = r2["id"]
        st, g = cl.get("/v1/responses/" + rid)
        check(st == 200 and g["id"] == rid and g["output"] == r2["output"], "GET a stored response")
        st, li = cl.get("/v1/responses/" + rid + "/input_items")
        check(st == 200 and [(i["type"], i.get("status")) for i in li["data"]] == [("function_call_output", "completed")],
              "its stored tool result")
        st, d = cl.delete("/v1/responses/" + rid)
        st2, _ = cl.get("/v1/responses/" + rid)
        check(st == 200 and d["deleted"] and st2 == 404, "DELETE removes it")
        st, _ = cl.post("/v1/responses", {"model": model, "previous_response_id": r3["id"], "input": "and?"})
        check(st == 400, "store: false keeps nothing to continue")

    # The event stream.
    types, seqs, text, reasoning, final, nested = [], [], "", "", None, True
    open_items = set()
    # The protocol whichever way the reply ends (as for chat completions above): "response.incomplete" where it ran out
    # of tokens, the answer checked where there is one.
    for ev, d in cl.stream("/v1/responses", {"model": model, "stream": True, "max_output_tokens": THINK_TOKENS,
                                             "temperature": 0, "input": "Is 91 a prime number? Answer yes or no."}):
        if ev == "http_error":
            check(False, "stream", str(d))
            return
        types.append(ev)
        seqs.append(d["sequence_number"])
        nested &= d["type"] == ev
        if ev == "response.output_item.added":
            open_items.add(d["output_index"])
        elif ev == "response.output_item.done":
            nested &= d["output_index"] in open_items
            open_items.discard(d["output_index"])
        elif ev.endswith(".delta"):
            nested &= d["output_index"] in open_items
        if ev == "response.output_text.delta":
            text += d["delta"]
        if ev == "response.reasoning_text.delta":
            reasoning += d["delta"]
        if ev in ("response.completed", "response.incomplete"):
            final = d["response"]
    cut = bool(types) and types[-1] == "response.incomplete"
    check(types[:2] == ["response.created", "response.in_progress"] and
          types[-1] in ("response.completed", "response.incomplete") and seqs == list(range(len(seqs))),
          "stream: lifecycle events, sequence numbers")
    check(nested and not open_items, "stream: deltas inside their items, every item done")
    if cut:
        print(f"  note stream: the response ran out of its {THINK_TOKENS} tokens; no answer to check")
        check(len(reasoning) > 20 and final is not None and final["status"] == "incomplete" and
              (final.get("incomplete_details") or {}).get("reason") == "max_output_tokens" and
              [o["type"] for o in final["output"]][:1] == ["reasoning"] and texts(final) == text,
              "stream: a response cut at its limit holds what streamed", json.dumps(final)[:300])
    else:
        check(len(reasoning) > 20 and "no" in text.lower(), "stream: reasoning and text deltas", text[:200])
        check(final is not None and [o["type"] for o in final["output"]] == ["reasoning", "message"] and
              texts(final) == text, "stream: the final response holds what streamed")

    # Rules.
    st, r = cl.post("/v1/responses", {"model": model, "input": "Hello there!", "tools": [fn], "tool_choice": "required",
                                      "temperature": 0, "max_output_tokens": 1024})
    calls = [o for o in r.get("output", []) if o["type"] == "function_call"] if st == 200 else []
    check(bool(calls) and fits_weather(json.loads(calls[0]["arguments"])), "tool_choice required",
          json.dumps(r)[:300])
    grouped = {"model": model, "input": "Get the weather in Paris.", "reasoning": {"effort": "none"},
               "tools": [fn, {"type": "namespace", "name": "forecast", "description": "Forecasts", "tools": [fn]}],
               "tool_choice": {"type": "function", "namespace": "forecast", "name": fn["name"]},
               "parallel_tool_calls": False, "temperature": 0, "max_output_tokens": 160}
    st, r = cl.post("/v1/responses", grouped)
    calls = [o for o in r.get("output", []) if o["type"] == "function_call"]
    check(st == 200 and len(calls) == 1 and (calls[0].get("namespace"), calls[0]["name"]) == ("forecast", fn["name"]),
          "namespaced function retains its wire identity", json.dumps(r)[:300])
    stream = list(cl.stream("/v1/responses", dict(grouped, stream=True)))
    items = [d["item"] for ev, d in stream if ev in ("response.output_item.added", "response.output_item.done")
             and d["item"]["type"] == "function_call"]
    done = [d for ev, d in stream if ev == "response.function_call_arguments.done"]
    check(len(items) == 2 and all(i.get("namespace") == "forecast" and i["name"] == fn["name"] for i in items)
          and len(done) == 1 and done[0]["name"] == fn["name"],
          "streamed call items and argument events use the wire name")
    st, r = cl.post("/v1/responses", {"model": model, "input": "Give me a person with a name and an age.",
                                      "reasoning": {"effort": "none"}, "temperature": 0, "max_output_tokens": 200,
                                      "text": {"format": {"type": "json_schema", "name": "person", "schema": {
                                          "type": "object", "properties": {"name": {"type": "string"},
                                                                           "age": {"type": "integer"}},
                                          "required": ["name", "age"], "additionalProperties": False}}}})
    try:
        person = json.loads(texts(r)) if st == 200 else None
    except ValueError:
        person = None
    check(isinstance(person, dict) and set(person) == {"name", "age"}, "text.format json_schema",
          texts(r)[:200] if st == 200 else json.dumps(r)[:200])
    st, r = cl.post("/v1/responses", {"model": model, "input": "hi", "tools": [{"type": "web_search"}]})
    check(st == 400 and "error" in r, "other tool types -> 400", json.dumps(r)[:200])


def concurrent_checks(cl: Client, model: str):
    """Two requests at once take turns on the GPU: the one sent later does not wait for
    the other to end, and each gets the reply it gets alone."""
    print("two requests at once")

    def body(prompt, n):
        return {"model": model, "stream": True, "max_tokens": n, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": prompt}]}
    long_body = body("Write a long story about a lighthouse keeper and the sea.", 700)
    short_body = body("List the first five prime numbers, separated by commas. Nothing else.", 32)

    def run(b, out, started=None):
        text = ""
        for ev, d in cl.stream("/v1/chat/completions", b):
            if ev == "http_error":
                out["error"] = str(d)
                break
            if d == "[DONE]":
                break
            for ch in d.get("choices", []):
                if ch["delta"].get("content"):
                    text += ch["delta"]["content"]
                    if started:
                        started.set()
        if started:
            started.set()
        out.update(text=text, end=time.time())

    st, t = cl.get("/v1/telemetry")
    sessions = t["model"].get("max_sessions", 0) if st == 200 else 0
    alone_short, alone_long, short, long = {}, {}, {}, {}
    run(short_body, alone_short)
    run(long_body, alone_long)
    writing = threading.Event()
    t_long = threading.Thread(target=run, args=(long_body, long, writing))
    t_long.start()
    writing.wait(120)  # the long reply is being written
    t_short = threading.Thread(target=run, args=(short_body, short))
    t_short.start()
    together = 0
    while t_short.is_alive():
        st, t = cl.get("/v1/telemetry")
        together = max(together, t["state"].get("running", 0) if st == 200 else 0)
        time.sleep(0.005)
    t_short.join()
    t_long.join()
    if sessions >= 2:
        check("error" not in short and "error" not in long and short.get("end", 0) < long.get("end", 0),
              "the request sent later ends first: it did not wait for the long one",
              f"short {short.get('error')}, long {long.get('error')}")
        check(together == 2, "telemetry counts two running requests", str(together))
    else:  # one session fits in this GPU's memory (the server says so at start): the two take turns whole
        print(f"  skip the request sent later ends first, two running requests: {sessions} session in GPU memory")
    check(bool(alone_short.get("text")) and short.get("text") == alone_short.get("text") and
          len(alone_long.get("text", "")) > 500 and long.get("text") == alone_long.get("text"),
          "each gets the reply it gets alone",
          f"short {short.get('text')!r} / {alone_short.get('text')!r}, long {len(long.get('text', ''))} chars")


def wait_telemetry(cl, predicate, seconds):
    end = time.monotonic() + seconds
    while True:
        _, t = cl.get("/v1/telemetry?recent=1")
        if predicate(t) or time.monotonic() >= end:
            return t
        time.sleep(0.02)


def superseded_response_checks(cl: Client, model: str):
    path = "/v1/responses"
    body = {"model": model, "input": "Spell out the numbers from 1 to 2000, one per line.",
            "max_output_tokens": 4096, "reasoning": {"effort": "none"}, "temperature": 0}
    for stream in (False, True):
        _, before = cl.get("/v1/telemetry?recent=0")
        count = before["totals"]["requests"]
        conn = cl._conn()
        try:
            conn.request("POST", path, json.dumps(dict(body, stream=stream)), cl._headers())
            mid = wait_telemetry(cl, lambda t: t["state"].get("generated", 0) >= 4 or
                                 t["totals"]["requests"] > count, 60)
            check(mid["state"].get("generated", 0) >= 4, "supersede a Responses request during decode")
            st, replacement = cl.post(path, dict(body, max_output_tokens=1))
            check(st == 200 and replacement.get("incomplete_details") == {"reason": "max_output_tokens"}
                  and replacement.get("completed_at") is None, "the replacement reports its own output limit")
            with conn.getresponse() as response:
                if stream:
                    events = [json.loads(line[6:]) for line in response if line.startswith(b"data: ")]
                    terminal = [e for e in events if e["type"] in ("response.completed", "response.failed",
                                                                   "response.incomplete", "error")]
                    check(terminal and terminal == events[-1:] and terminal[0]["type"] == "response.incomplete",
                          "a superseded stream ends with one incomplete event")
                    old = events[-1].get("response", {}) if events else {}
                else:
                    old = json.load(response)
        finally:
            conn.close()
        check(old.get("status") == "incomplete" and old.get("incomplete_details") is None and
              old.get("completed_at") is None and old.get("error") is None and
              old.get("output") and old["output"][-1].get("status") == "incomplete" and
              0 < old["usage"]["output_tokens"] < body["max_output_tokens"],
              "the superseded response keeps its partial output and usage")
        st, stored = cl.get(path + "/" + old.get("id", "unknown"))
        check(st == 200 and stored == old, "GET returns the same partial response")
        _, telemetry = cl.get("/v1/telemetry?recent=2")
        replaced = [r for r in telemetry["recent"] if r.get("cancel_reason") == "superseded"]
        check(len(replaced) == 1 and replaced[0]["status"] == "cancelled" and
              replaced[0]["output_tokens"] == old["usage"]["output_tokens"], "telemetry explains the partial response")
        st, log = cl.get("/v1/requests?limit=2")
        if st == 200 and log.get("enabled"):
            entry = next((r for r in log["requests"] if r["id"] == old["id"]), {})
            check(entry.get("status") == "cancelled" and entry.get("cancel_reason") == "superseded",
                  "the request log records the replacement")


def nonstream_disconnect_checks(cl: Client, model: str):
    """A caller waiting for JSON can disconnect while the scheduler is continuously emitting tokens."""

    messages = [{"role": "user", "content": "Write the numbers from 1 to 2000, one per line, each spelled out."}]
    for api, label, body in (("chat/completions", "chat", {"messages": messages, "max_tokens": 4096}),
                             ("responses", "responses", {"input": messages, "max_output_tokens": 4096, "store": False}),
                             ("messages", "messages", {"messages": messages, "max_tokens": 4096})):
        _, before = cl.get("/v1/telemetry?recent=0")
        count = before["totals"]["requests"]
        conn = cl._conn()
        try:
            conn.request("POST", "/v1/" + api, json.dumps(dict(body, model=model, temperature=0)), cl._headers())
            mid = wait_telemetry(cl, lambda t: t["state"].get("generated", 0) > 0 or t["totals"]["requests"] > count, 60)
        finally:
            conn.close()
        check(mid["state"].get("label") == label and mid["state"].get("generated", 0) > 0,
              api + ": disconnect during active nonstreaming generation")
        end = wait_telemetry(cl, lambda t: t["totals"]["requests"] > count, 10)
        last = end["recent"][0] if end["recent"] else {}
        check(last.get("label") == label and last.get("status") == "cancelled" and "cancel_reason" not in last
              and end["totals"]["cancelled"] == before["totals"]["cancelled"] + 1,
              api + ": the disconnected request stopped", json.dumps(last))


def telemetry_checks(cl: Client, model: str):
    """GET /v1/telemetry (what `flyx monitor` draws): mid-reply and after the checks above."""
    print("telemetry")
    mid = None
    for ev, d in cl.stream("/v1/chat/completions", {
            "model": model, "stream": True, "max_tokens": 1500, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "Write a long story about a lighthouse keeper."}]}):
        if ev == "http_error":
            check(False, "stream", str(d))
            return
        if d != "[DONE]" and any(ch["delta"].get("content") for ch in d.get("choices", [])) and mid is None:
            st, mid = cl.get("/v1/telemetry")
            break
    s = (mid or {}).get("state", {})
    check(s.get("phase") == "decode" and s.get("generated", 0) > 0 and s.get("label") == "chat" and
          mid["rates"]["decode_tok_s"] >= 0, "mid-reply: decoding, tokens so far", json.dumps(s))
    time.sleep(1.5)  # the closed stream is cancelled; its second is over
    st, t = cl.get("/v1/telemetry?recent=100")
    check(st == 200 and t.get("schema_version") == 1 and t["model"]["loaded"] and t["model"]["id"] == model,
          "schema 1, the model", json.dumps(t.get("model"))[:300])
    required = t["server"].get("api_key_required")
    check(isinstance(required, bool) and (not required or bool(cl.key)), "server reports its API key requirement")
    check(t["state"]["phase"] == "idle" and t["state"]["queued"] == 0, "idle again", json.dumps(t["state"]))
    rec = t["recent"]
    labels = {r["label"] for r in rec}
    check(len(rec) >= 20 and {"chat", "responses", "messages", "completion"} <= labels and
          rec[0]["status"] == "cancelled" and t["totals"]["requests"] >= len(rec) and
          t["totals"]["tool_calls"] > 0 and t["totals"]["reused_tokens"] > 0,
          "recent requests of every API, newest first; totals", f"{len(rec)} records, {sorted(labels)}")
    check(len(t["rates"]["output_per_second"]) == 60 and sum(t["rates"]["output_per_second"]) > 0,
          "output per second over the last minute")
    check(len(t["rates"]["prefill_per_second"]) == 60 and sum(t["rates"]["prefill_per_second"]) > 0,
          "prompt tokens computed per second over the last minute")
    r = t["rates"]
    check(all(len(r[k]) == 60 for k in ("rounds_per_second", "drafted_per_second", "accepted_per_second")) and
          sum(r["rounds_per_second"]) > 0 and sum(r["drafted_per_second"]) >= sum(r["accepted_per_second"]) >= 0,
          "rounds verified, drafts and drafts accepted per second over the last minute")
    check(all(0 <= x["accepted_drafts"] <= x["drafted"] for x in t["recent"]), "each request's drafts and drafts accepted")
    runtime = t["memory"].get("runtime", -1)
    check(runtime is None or runtime >= 0, "flyx's own runtime memory, where the system counts it", str(runtime))
    proc = t["process"]
    check(proc["ram"] >= proc["rss"] and proc["ram"] >= t["memory"]["host_mapped"] >= 0,
          "the host memory flyx holds in all: at least its resident memory and the K/V's mapped host memory",
          json.dumps(proc))
    d = t["device"]
    check(d is None or (0 <= d["util"] <= 100 and d["mem_total"] > d["mem_used"] > 0 and d["power_w"] > 0),
          "device figures", json.dumps(d))
    check(t["memory"]["sessions"] and t["memory"]["ledger_used"] > 0 and t["process"]["rss"] > 0 and
          t["system"]["ram_total"] > 0 and t["system"]["cpus"] > 0 and t["system"]["cpu_s"] >= 0,
          "sessions, memory, process, system")


def dashboard_checks(cl: Client):
    """GET /: the dashboard to a browser, the endpoints as JSON to anything else; GET /dashboard: the same page; its
    files (and the libraries it imports) under /dashboard/, as scripts; nothing outside them."""
    print("dashboard")

    def fetch(path, accept="*/*"):
        c = cl._conn()
        c.request("GET", path, headers={"Accept": accept})
        r = c.getresponse()
        return r.status, r.getheader("Content-Type", ""), r.read().decode("utf-8", "replace")

    for path, accept in (("/", "text/html,application/xhtml+xml,*/*;q=0.8"), ("/dashboard", "*/*"), ("/dashboard/", "*/*")):
        st, ct, body = fetch(path, accept)
        check(st == 200 and ct.startswith("text/html") and "/dashboard/app.js" in body and body.rstrip().endswith("</html>"),
              f"GET {path} for {accept.split(',')[0]}: the page", f"{st} {ct}")
    for path in ("/dashboard/app.js", "/dashboard/lib/telemetry.js", "/dashboard/vendor/preact.module.js",
                 "/dashboard/vendor/hooks.module.js", "/dashboard/vendor/htm.module.js"):
        st, ct, body = fetch(path)
        check(st == 200 and ct.startswith("text/javascript") and len(body) > 100, f"GET {path}: a script", f"{st} {ct}")
    st, ct, _ = fetch("/dashboard/style.css")
    check(st == 200 and ct.startswith("text/css"), "GET /dashboard/style.css: the stylesheet", f"{st} {ct}")
    for path in ("/dashboard/nothing.js", "/dashboard/../CMakeLists.txt", "/dashboard/%2e%2e/CMakeLists.txt"):
        st, _, _ = fetch(path)
        check(st in (400, 404), f"GET {path}: refused", str(st))
    st, j = cl.get("/")
    check(st == 200 and "/dashboard" in j.get("endpoints", []) and "/v1/telemetry" in j.get("endpoints", []),
          "GET / for a script: the endpoints as JSON", json.dumps(j)[:200])
    # The Integration reference lists every endpoint the server names, so the two cannot drift apart.
    _, _, docs = fetch("/dashboard/pages/integration.js")
    undocumented = [e for e in j.get("endpoints", []) if f'<span class="path">{e}' not in docs]
    check(not undocumented, "the Integration reference names every endpoint", f"missing: {undocumented}")


def model_panel_checks(cl: Client, model: str):
    """The dashboard's model panel: GET /v1/load names what is served; POST /v1/load refuses a body
    without a model before it fetches or loads anything; telemetry carries the same state; the cache's models."""
    print("model panel")
    st, s = cl.get("/v1/load")
    check(st == 200 and s["state"] == "ready" and s["model"] == model and s.get("error") is None,
          "GET /v1/load: serving the model", json.dumps(s)[:200])
    for body in ({}, {"model": ""}, {"model": 7}, {"model": "x/y", "ctx": -1}, {"model": "x/y", "drafter": "nope"},
                 {"model": "x/y", "drafters": "z-lab/a"},
                 {"model": "x/y", "width": -1}, {"model": "x/y", "draft_head": "nope"}, {"model": "x/y", "draft_weights": 7},
                 {"model": "x/y", "draft_weights": "nope"}):
        st, j = cl.post("/v1/load", body)
        check(st == 400 and "error" in j, f"POST /v1/load {json.dumps(body)}: 400", f"{st} {json.dumps(j)[:120]}")
    st, t = cl.get("/v1/telemetry")
    check(st == 200 and t["load"]["state"] == "ready" and t["model"]["id"] == model, "telemetry's load: the same")
    check(st == 200 and isinstance(s.get("drafters"), list) and isinstance(s.get("suggested"), list) and
          isinstance(s.get("degraded"), list) and
          "fallback" in s and all({"kind", "from", "size"} <= set(d) for d in s.get("drafters", [])),
          "GET /v1/load: drafters, suggested, degraded, fallback", json.dumps({k: s.get(k) for k in ("drafters", "suggested", "degraded")}))
    st, c = cl.get("/v1/hub/cached")
    check(st == 200 and isinstance(c.get("models"), list) and c.get("cache"), "GET /v1/hub/cached: a list",
          json.dumps(c)[:200])
    if st != 200:
        return
    fields = {"name", "kind", "source", "repo", "quant", "file", "path", "size", "modified", "revision", "current", "mtp"}
    check(c.get("cache_from") in ("HF_HUB_CACHE", "HF_HOME", "default") and isinstance(c.get("dirs"), list) and
          all(fields <= set(m) and m["kind"] in ("gguf", "flyx", "drafter") and m["source"] in ("hub", "dir") and
              m["mtp"] in ("in-file", "in-repo", None) for m in c["models"]),
          "GET /v1/hub/cached: every field of every model", json.dumps(c["models"][:2])[:300])
    names = [m["name"] for m in c["models"]]
    check(len(names) == len(set(names)), "GET /v1/hub/cached: each name once")
    entry = served_entry(c, s)
    if entry is None:
        print("  note the served model is not in the cache or a model directory: its entry is not checked")
    else:
        check(entry["current"] is True, "the served model is listed, current", json.dumps(entry)[:300])
        st, j = cl.delete("/v1/hub/cached?name=" + urllib.parse.quote(entry["name"], safe=""))
        check(st == 409 if entry["source"] == "hub" else st == 400,
              "DELETE /v1/hub/cached for the served model: refused", f"{st} {json.dumps(j)[:200]}")
    st, j = cl.delete("/v1/hub/cached?name=" + urllib.parse.quote("acme/no-such-model-GGUF:Q4_0", safe=""))
    check(st in (400, 404) and "error" in j, "DELETE /v1/hub/cached for a name not listed: refused", f"{st}")
    hub_dir_checks(cl, fields, (s.get("loaded") or {}).get("model") or "")


def hub_dir_checks(cl: "Client", fields: set, served: str) -> None:
    """GET /v1/hub/dir: where to start from, a folder's folders and model files, a model file by itself."""
    st, j = cl.get("/v1/hub/dir")
    check(st == 200 and j.get("kind") == "start" and j.get("parent") is None and j.get("models") == [] and
          all({"name", "path"} <= set(d) for d in j.get("dirs", [])), "GET /v1/hub/dir: where to start from", json.dumps(j)[:300])
    st, j = cl.get("/v1/hub/dir?path=" + urllib.parse.quote(os.path.join(os.getcwd(), "no", "such", "folder"), safe=""))
    check(st == 404 and "error" in j, "GET /v1/hub/dir for no such folder: 404", f"{st}")
    if not served or not os.path.isfile(served) or not served.lower().endswith((".gguf", ".flyx")):
        # A Hub name is not a file; a file of an older extension loads but is not among what a folder lists.
        print("  note the served model is not a .gguf or .flyx file of this machine: its folder is not browsed")
        return
    path = os.path.abspath(served)
    st, j = cl.get("/v1/hub/dir?path=" + urllib.parse.quote(os.path.dirname(path), safe=""))
    same = lambda a, b: os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
    check(st == 200 and j.get("kind") == "dir" and same(j.get("path", ""), os.path.dirname(path)) and
          all(fields <= set(m) and m["source"] == "dir" for m in j.get("models", [])) and
          any(same(m["path"], path) for m in j.get("models", [])),
          "GET /v1/hub/dir for the served model's folder: lists it", json.dumps(j)[:400])
    st, j = cl.get("/v1/hub/dir?path=" + urllib.parse.quote(path, safe=""))
    check(st == 200 and j.get("kind") == "file" and len(j.get("models", [])) == 1 and same(j["models"][0]["path"], path) and
          same(j.get("parent", ""), os.path.dirname(path)), "GET /v1/hub/dir for a model file: that file", json.dumps(j)[:400])


def served_entry(cached: dict, status: dict):
    """The listing's entry for the model served: by the name it was loaded with, else by its file's path."""
    want = (status.get("loaded") or {}).get("model") or ""
    path = os.path.normcase(os.path.abspath(want)) if want and os.path.exists(want) else None
    for m in cached.get("models", []):
        if m["name"].lower() == want.lower() or (path and os.path.normcase(m["path"]) == path):
            return m
    return None


def wait_load(cl: Client, timeout_s: float) -> dict:
    """GET /v1/load until no fetch or load is under way, at most timeout_s."""
    deadline = time.time() + timeout_s
    s = {}
    while time.time() < deadline:
        st, s = cl.get("/v1/load")
        if st == 200 and s.get("state") not in ("fetching", "loading"):
            return s
        time.sleep(1)
    return s


def chat_status(cl: Client, model: str) -> int:
    st, _ = cl.post("/v1/chat/completions", {"model": model, "max_tokens": 8, "temperature": 0,
                                             "messages": [{"role": "user", "content": "Say ok."}],
                                             "chat_template_kwargs": {"enable_thinking": False}})
    return st


def failed_switch_checks(cl: Client, model: str):
    """a model that fails to load stays unloaded in state 'failed' (no auto-reload of the previous model)."""
    print("a failed switch stays unloaded")
    st_b, before = cl.get("/v1/load")
    remembered = before.get("remembered") if st_b == 200 else None
    bogus = os.path.join(tempfile.mkdtemp(prefix="flyx_check_"), "bogus.gguf")
    with open(bogus, "wb") as f:
        f.write(b"not a model at all \x00\x01\x02\xff")
    st, s = cl.post("/v1/load", {"model": bogus})
    if st == 403:
        print("  note POST /v1/load: 403 (not this machine and no --key): failed switch is not checked")
        return
    check(st == 202, "POST /v1/load of a bogus file: started", f"{st} {json.dumps(s)[:200]}")
    s = wait_load(cl, 180)
    check(s.get("state") == "failed" and s.get("fallback") is None and s.get("model") is None and
          bool(s.get("error")), "it fails without auto-reloading the old model", json.dumps(s)[:300])
    check(chat_status(cl, model) == 503, "a chat request after a failed switch: 503")
    os.remove(bogus)
    check(bool(remembered), "the settings file still remembers the previous model", json.dumps(before)[:200])
    if not remembered:
        return
    st, s = cl.post("/v1/load", remembered)
    check(st == 202, "POST /v1/load of the remembered model after failed switch", f"{st} {json.dumps(s)[:200]}")
    s = wait_load(cl, 300)
    check(s.get("state") == "ready" and s.get("model") == model, "served again after explicit reload", json.dumps(s)[:200])
    check(chat_status(cl, model) == 200, "a chat request after explicit reload")


def unload_checks(cl: Client, model: str):
    """POST /v1/unload: none served, requests get 503; then the remembered model loaded again, as the run began."""
    print("unload")
    st, before = cl.get("/v1/load")
    remembered = before.get("remembered") if st == 200 else None
    st, s = cl.post("/v1/unload", {})
    if st == 403:
        print("  note POST /v1/unload: 403 (not this machine and no --key): not checked")
        return
    check(st == 200 and s.get("state") == "none" and s.get("model") is None, "POST /v1/unload: none served",
          f"{st} {json.dumps(s)[:200]}")
    check(chat_status(cl, model) == 503, "a chat request with none served: 503")
    check(bool(remembered), "the settings file still names the model", json.dumps(before)[:200])
    if not remembered:
        return
    st, s = cl.post("/v1/load", remembered)
    check(st == 202, "POST /v1/load of the remembered model", f"{st} {json.dumps(s)[:200]}")
    s = wait_load(cl, 300)
    check(s.get("state") == "ready" and s.get("model") == model, "served again", json.dumps(s)[:200])


def request_timing_checks(cl: Client, model: str):
    """Two Messages requests of different lengths finish near each other. Each log entry owns its engine times."""
    path = "/v1/messages"
    head = {"model": model, "thinking": {"type": "disabled"}, "temperature": 0}
    main = dict(head, messages=[{"role": "user", "content": "Write the integers from 1 to 2000, one per line."}], max_tokens=128)
    title = dict(head, messages=[{"role": "user", "content": "Give a short title for a book about trees."}], max_tokens=1)
    conn = cl._conn()
    try:
        _, before = cl.get("/v1/telemetry?recent=0")
        conn.request("POST", path, json.dumps(main), cl._headers())
        mid = wait_telemetry(cl, lambda t: t["state"].get("generated", 0) >= 4 or
                             t["totals"]["requests"] > before["totals"]["requests"], 60)
        check(mid["state"].get("generated", 0) >= 4, "the longer Messages request has started writing")
        st, short = cl.post(path, title)
        with conn.getresponse() as response:
            long = json.load(response)
            check(st == response.status == 200, "both Messages requests finished")
    finally:
        conn.close()
    _, log = cl.get("/v1/requests?limit=2")
    _, telemetry = cl.get("/v1/telemetry?recent=2")
    for reply in (short, long):
        row = next((r for r in log["requests"] if r["id"] == reply["id"]), {})
        output = reply["usage"]["output_tokens"]
        rec = next((r for r in telemetry["recent"] if r["output_tokens"] == output), {})
        timing = row.get("timing", {})
        ttft = row.get("ttft_ms")
        check(bool(rec) and bool(timing) and row["usage"]["output_tokens"] == output and
              all(timing[k] == rec[k] for k in ("queue_ms", "prefill_ms", "decode_ms")) and
              (ttft is None or ttft >= timing["queue_ms"]) and
              (timing["decode_tok_s"] or 0) == rec["decode_tok_s"] and
              (timing["tokens_per_round"] or 0) == rec["tokens_per_round"],
              "the Messages entry keeps its own timings for " + str(output) + " output tokens")


def request_log_checks(cl: Client, model: str):
    """The request log: a chat request, plain and streamed, is in GET /v1/requests with its prompt
    and answer."""
    print("request log")
    tag = f"reqlog-{time.time_ns()}"  # a prompt no other check sends, to find its entry by

    def ask(stream: bool) -> str:
        prompt = f"Reply with the single word ok. ({tag}-{'stream' if stream else 'plain'})"
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 16,
                "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
        if not stream:
            st, r = cl.post("/v1/chat/completions", body)
            check(st == 200, "a chat request", json.dumps(r)[:200])
            return prompt
        for _, d in cl.stream("/v1/chat/completions", dict(body, stream=True)):
            if d == "[DONE]":
                break
        return prompt

    def logged(prompt: str):
        """The log's entries, and the one whose request carries `prompt`.  A stream's entry is written as its last
        event goes out, so it may come a moment after the client has read it."""
        for _ in range(20):
            st, j = cl.get("/v1/requests")
            if st != 200:
                return st, j, None
            for e in j.get("requests", []):
                if any(m.get("content") == prompt for m in (e.get("request") or {}).get("messages", [])):
                    return st, j, e
            time.sleep(0.1)
        return st, j, None

    plain = ask(False)
    st, j, e = logged(plain)
    if st == 403:
        print("  note GET /v1/requests: 403 (not this machine and no --key): the log is not checked")
        return
    check(st == 200 and j.get("enabled") is True, "GET /v1/requests: enabled", json.dumps(j)[:200])
    newest = (j.get("requests") or [{}])[0]
    check(e is not None and newest is e, "the plain request is the newest entry", json.dumps(newest)[:300])
    e = e or {}
    _, tm = cl.get("/v1/telemetry")
    matches = [r for r in tm.get("recent", []) if r.get("id") == e.get("id") and r["label"] == "chat"]
    check(len(matches) == 1 and matches[0]["output_tokens"] == e.get("usage", {}).get("output_tokens"),
          "Chat telemetry identifies the same completion as its request record")
    check(e.get("api") == "chat" and e.get("model") == model and e.get("stream") is False and
          e.get("status") in ("stop", "length") and e.get("finished_at", 0) >= e.get("started_at", 1),
          "its api, model, status and times", json.dumps({k: v for k, v in e.items() if k != "request"})[:300])
    check(bool(e.get("content")) and e.get("usage", {}).get("output_tokens", 0) > 0, "its answer and usage",
          json.dumps({k: e.get(k) for k in ("content", "usage")}))
    path = "/v1/requests?limit=1"
    st, h, raw = cl.get_raw(path)
    tag = h.get("ETag", "")
    check(st == 200 and bool(tag) and json.loads(raw)["requests"] == [e], "a validator for the selected history")
    st, h, raw = cl.get_raw(path, {"If-None-Match": '"other,tag", W/' + tag})
    check(st == 304 and h.get("ETag") == tag and not raw, "unchanged history has no response body")
    st, _, raw = cl.get_raw("/v1/requests?limit=0", {"If-None-Match": tag})
    check(st == 200 and json.loads(raw)["requests"] == [], "changing the selected rows invalidates the validator")
    streamed = ask(True)
    st, j, e = logged(streamed)
    check(e is not None and e.get("api") == "chat" and e.get("stream") is True and bool(e.get("content")) and
          e.get("usage", {}).get("output_tokens", 0) > 0, "a streamed request: logged with its answer",
          json.dumps({k: v for k, v in (e or {}).items() if k != "request"})[:300])
    st, h, raw = cl.get_raw(path, {"If-None-Match": tag})
    check(st == 200 and h.get("ETag") and h["ETag"] != tag and json.loads(raw)["requests"] == [e],
          "new entries invalidate the validator")
    request_timing_checks(cl, model)

    for index, (headers, expected) in enumerate((
        ({"x-claude-code-request-class": "compaction", "x-claude-code-compaction": "manual",
          "x-claude-code-session-id": "private-session"}, {"request_class": "compaction", "compaction": "manual"}),
        ({"x-claude-code-request-class": "main", "x-claude-code-context-compacted": "reactive"},
         {"request_class": "main", "context_compacted": "reactive"}),
        ({"x-claude-code-request-class": "future-class", "x-claude-code-compaction": "unrecognized",
          "x-claude-code-context-compacted": "arbitrary"}, None),
        ({}, None),
    )):
        prompt = f"Gateway hint {index} ({tag})"
        msg = {"model": model, "max_tokens": 0, "messages": [{"role": "user", "content": prompt}]}
        st, _ = cl.post("/v1/messages", msg, headers)
        _, _, entry = logged(prompt)
        check(st == 200 and entry and entry.get("client_hints") == expected and "private-session" not in json.dumps(entry),
              f"Messages records only known gateway hints (case {index + 1})")

    prompt = f"Check the byte count in the request log. ({tag})"
    original = "中" * 26667
    st, _ = cl.post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1, "chat_template_kwargs": {"log_test": [original] * 5}})
    _, _, entry = logged(prompt)
    parts = (entry or {}).get("request", {}).get("chat_template_kwargs", {}).get("log_test", [])
    clipped = parts[0] if parts else ""
    prefix, marker, omitted = clipped.partition("… [truncated ")
    expected = len(original.encode()) - len(prefix.encode())
    check(st == 200 and len(parts) == 5 and marker and original.startswith(prefix) and
          omitted == f"{expected} bytes]", "a multi-pass log cut counts the original UTF-8 bytes")

    # These 10 values fit in 256 KiB as text at 16 KiB apiece. JSON escapes put the serialized body over that
    # target, so the log chooses 4 KiB. Check the insertion order of the keys as well as the retained bytes.
    prompt = f"Check escaped request text. ({tag})"
    text = '\"\\\n\t' * 5100 + "終"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1,
            "log_probe": [{"z": i, "a": text, "empty": {}, "flag": False} for i in range(10)]}
    st, _ = cl.post("/v1/chat/completions", body)
    _, _, entry = logged(prompt)
    kept = text[:4096] + f"… [truncated {len(text.encode()) - 4096} bytes]"
    for item in body["log_probe"]:
        item["a"] = kept
    check(st == 200 and entry and json.dumps(entry["request"]) == json.dumps(body),
          "escaped length determines the cap and every request key retains its position")

    prompt = f"Check a request with many short values. ({tag})"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1,
            "log_probe": [{"name": "call", "arguments": "x" * 900} for _ in range(320)]}
    st, _ = cl.post("/v1/chat/completions", body)
    _, _, entry = logged(prompt)
    check(st == 200 and entry and entry["request"] == body,
          "all short tool arguments remain when their combined size exceeds 256 KiB")


def system_log_checks(cl: Client):
    """The system log: GET /v1/logs has the server's own lines (failed_switch_checks, run before, made it
    write that a model did not load), numbered in order, and gives only what came after a number."""
    print("system log")
    st, j = cl.get("/v1/logs?limit=2000")
    if st == 403:
        print("  note GET /v1/logs: 403 (not this machine and no --key): the log is not checked")
        return
    check(st == 200 and j.get("enabled") is True, "GET /v1/logs: enabled", json.dumps(j)[:200])
    lines = j.get("lines", [])
    seqs = [l.get("seq") for l in lines]
    check(bool(lines) and seqs == sorted(seqs) and len(set(seqs)) == len(seqs) and j.get("next") == seqs[-1],
          "lines numbered in order, next is the last", json.dumps(j)[:300])
    check(all(isinstance(l.get("t"), (int, float)) and isinstance(l.get("text"), str) for l in lines), "each has t and text")
    check(any("did not load" in l.get("text", "") for l in lines), "the failed load (the failed switch check) is in the log",
          json.dumps([l.get("text") for l in lines[-8:]])[:400])
    st, k = cl.get(f"/v1/logs?after={j['next']}")
    check(st == 200 and all(l["seq"] > j["next"] for l in k.get("lines", [])), "?after=N: only the lines past N")
    st, k = cl.get("/v1/logs?after=999999999")
    check(st == 200 and k.get("lines") == [] and k.get("next", 0) <= 999999999, "?after past the end: nothing, next is the last kept")
    st, k = cl.get("/v1/logs?limit=1")
    check(st == 200 and len(k.get("lines", [])) == 1, "?limit=1: one line")


def anthropic_prewarm_checks(cl: Client, model: str):
    body = {"model": model, "max_tokens": 0, "temperature": 0,
            "system": [{"type": "text", "text": "Keep these local reference notes. " * 200,
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": "Read the reference notes."}]}
    st, count = cl.post("/v1/messages/count_tokens", body)
    check(st == 200 and count.get("input_tokens", 0) > 0, "count the prompt before prewarming")
    st, msg = cl.post("/v1/messages", body)
    usage = msg.get("usage", {})
    check(st == 200 and msg.get("content") == [] and msg.get("stop_reason") == "max_tokens" and
          usage.get("output_tokens") == 0 and usage["input_tokens"] + usage["cache_read_input_tokens"] == count["input_tokens"],
          "prewarm returns the counted prompt with zero output")
    st, replay = cl.post("/v1/messages", dict(body, max_tokens=4))
    check(st == 200 and replay.get("usage", {}).get("output_tokens", 0) > 0 and
          replay["usage"]["cache_read_input_tokens"] > 0, "positive token budgets generate from the warm prompt")
    tool = {"name": "now", "input_schema": {"type": "object"}}
    for option in ({"stream": True}, {"thinking": {"type": "enabled", "budget_tokens": 1024}},
                   {"output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}}},
                   {"tools": [tool], "tool_choice": {"type": "any"}},
                   {"tools": [tool], "tool_choice": {"type": "tool", "name": "now"}}):
        st, reply = cl.post("/v1/messages", {**body, **option})
        check(st == 400 and reply.get("error", {}).get("type") == "invalid_request_error",
              "incompatible prewarm option returns 400: " + str(option))


def anthropic_schema_checks(cl: Client, model: str):
    body = {"model": model, "messages": [{"role": "user", "content": "Are you ready?"}],
            "max_tokens": 100, "temperature": 0, "thinking": {"type": "disabled"},
            "output_config": {"format": {"type": "json_schema", "schema": {"type": "string", "enum": ["ready"]}}}}
    st, count = cl.post("/v1/messages/count_tokens", body)
    check(st == 200 and count.get("input_tokens", 0) > 0, "count a request with a structured output format")
    st, msg = cl.post("/v1/messages", body)
    text = "".join(b.get("text", "") for b in msg.get("content", []))
    check(st == 200 and msg.get("stop_reason") == "end_turn" and text.strip() == '"ready"',
          "structured output returns the JSON value allowed by the schema", text[:80])
    events = list(cl.stream("/v1/messages", dict(body, stream=True)))
    text = "".join(e["delta"].get("text", "") for kind, e in events if kind == "content_block_delta")
    stop = [e["delta"].get("stop_reason") for kind, e in events if kind == "message_delta"]
    check(text.strip() == '"ready"' and stop == ["end_turn"], "streamed text forms the same JSON value", text[:80])
    assistant = [*body["messages"], {"role": "assistant", "content": "Here is the answer:"}]
    for change in ({"messages": assistant}, {"output_config": {"format": {"type": "text"}}},
                   {"output_config": {"format": {"type": "json_schema", "schema": []}}}):
        st, msg = cl.post("/v1/messages", {**body, **change})
        check(st == 400 and msg.get("error", {}).get("type") == "invalid_request_error",
              "invalid structured output request returns 400: " + str(change))


def anthropic_replacement_checks(cl: Client, model: str):
    path = "/v1/messages"
    original = {"model": model, "messages": [{"role": "user", "content": "Spell out the numbers from 1 to 2000."}],
                "thinking": {"type": "disabled"}, "max_tokens": 4096, "temperature": 0}
    for stream in (False, True):
        _, before = cl.get("/v1/telemetry?recent=0")
        count = before["totals"]["requests"]
        conn = cl._conn()
        try:
            conn.request("POST", path, json.dumps(dict(original, stream=stream)), cl._headers())
            mid = wait_telemetry(cl, lambda t: t["state"].get("generated", 0) >= 4 or
                                 t["totals"]["requests"] > count, 60)
            check(mid["state"].get("generated", 0) >= 4, "replace an active Messages request")
            st, new = cl.post(path, dict(original, max_tokens=1))
            check(st == 200 and new.get("usage", {}).get("output_tokens") == 1, "the replacement generates normally")
            with conn.getresponse() as old:
                if stream:
                    events = [json.loads(line[6:]) for line in old if line.startswith(b"data: ")]
                    kinds = [e["type"] for e in events]
                    check(old.status == 200 and "content_block_delta" in kinds and kinds[-1:] == ["message_stop"]
                          and "error" not in kinds, "a replacement keeps the existing Messages stream terminal event")
                else:
                    err = json.load(old).get("error", {})
                    check(old.status == 409 and err.get("type") == "conflict_error" and
                          old.getheader("x-should-retry") == "false", "a replaced Messages request advises against retry")
        finally:
            conn.close()


def anthropic_checks(cl: Client, model: str):
    print("Anthropic /v1/messages")
    anthropic_replacement_checks(cl, model)
    anthropic_prewarm_checks(cl, model)
    anthropic_schema_checks(cl, model)
    tools = [{"name": "get_weather", "description": "Get the current weather for a city.",
              "input_schema": WEATHER["function"]["parameters"]}]
    body = {"model": model, "max_tokens": 1024, "temperature": 0, "tools": tools,
            "system": [{"type": "text", "text": "You are a helpful assistant."}],
            "messages": [{"role": "user", "content": "What's the weather in Tokyo? Use the tool."}]}
    st, r = cl.post("/v1/messages", body)
    blocks = r.get("content", []) if st == 200 else []
    uses = [b for b in blocks if b["type"] == "tool_use"]
    check(st == 200 and r["stop_reason"] == "tool_use" and uses and uses[0]["name"] == "get_weather" and
          "tokyo" in str(uses[0]["input"].get("city", "")).lower(), "tool_use block", json.dumps(r)[:400])
    check(any(b["type"] == "thinking" for b in blocks), "thinking block")
    if uses:
        body["messages"] += [
            {"role": "assistant", "content": blocks},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": uses[0]["id"],
                                          "content": [{"type": "text", "text": "22 C, sunny"}]}]}]
        # streaming follow-up
        text, thinking, types, stop, usage, start_usage = "", "", [], None, None, {}
        for ev, d in cl.stream("/v1/messages", dict(body, stream=True)):
            if ev == "http_error":
                check(False, "stream", str(d))
                break
            types.append(ev)
            if ev == "message_start":
                start_usage = d["message"]["usage"]
            if ev == "content_block_delta":
                text += d["delta"].get("text", "")
                thinking += d["delta"].get("thinking", "")
            if ev == "message_delta":
                stop, usage = d["delta"]["stop_reason"], d["usage"]
        check("22" in text, "stream: answer uses the tool result", text[:200])
        check(types[0] == "message_start" and types[-1] == "message_stop" and stop == "end_turn",
              "stream: event order and stop_reason", " ".join(types[:6]) + " ... " + " ".join(types[-3:]))
        check(usage is not None and usage["cache_read_input_tokens"] > 0, "stream: reused prompt tokens reported",
              json.dumps(usage))
        check(usage is not None and start_usage == {**usage, "output_tokens": 0},
              "stream: the start and final events agree on new and cached input", json.dumps(start_usage))
    st, r = cl.post("/v1/messages", dict(body, tool_choice={"type": "any"},
                                         messages=[{"role": "user", "content": "Hello there!"}]))
    uses = [b for b in r.get("content", []) if b["type"] == "tool_use"] if st == 200 else []
    check(st == 200 and r["stop_reason"] == "tool_use" and uses and fits_weather(uses[0]["input"]),
          "tool_choice any: a tool_use even for a greeting", json.dumps(r)[:300])
    # Claude Code's billing line changes on every request; left out, the next turn continues the
    # session instead of computing the whole prompt again.
    rules = "You are a coding agent working in a large repository. Keep answers short. " * 40

    def billed(cch, messages):
        return {"model": model, "max_tokens": 8, "temperature": 0, "thinking": {"type": "disabled"},
                "system": [{"type": "text", "text": f"x-anthropic-billing-header: cc_version=2.1.3; cch={cch};"},
                           {"type": "text", "text": rules}],
                "messages": messages}
    turn1 = [{"role": "user", "content": "Name a prime number between 10 and 20."}]
    st1, r1 = cl.post("/v1/messages", billed("a1b2c", turn1))
    reply = "".join(b.get("text", "") for b in r1.get("content", [])) if st1 == 200 else ""
    first = r1["usage"]["input_tokens"] + r1["usage"]["cache_read_input_tokens"] if st1 == 200 else 1 << 30
    turn2 = turn1 + [{"role": "assistant", "content": reply}, {"role": "user", "content": "And one between 20 and 30?"}]
    st2, r = cl.post("/v1/messages", billed("9f8e7", turn2))
    u = r.get("usage", {}) if st2 == 200 else {}
    check(st2 == 200 and u.get("cache_read_input_tokens", 0) >= first - 64,
          f"a changed Claude Code billing line keeps the session ({first}-token first turn)", json.dumps(u))
    st, r = cl.post("/v1/messages/count_tokens", {"model": model, "messages": [{"role": "user", "content": "hi"}]})
    check(st == 200 and r.get("input_tokens", 0) > 0, "count_tokens", json.dumps(r))
    st, r = cl.post("/v1/messages", {"model": model, "max_tokens": 64,
                                     "messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]})
    check(st == 400 and r.get("type") == "error", "unsupported content -> 400 error")


def context_checks(cl: Client, model: str):
    """A prompt longer than the context: an error status before any stream, worded as each API words it (what clients
    look for to shorten the conversation); the next request is served as ever."""
    print("A prompt longer than the context")
    st, m = cl.get("/v1/models")
    ctx = m["data"][0].get("max_model_len", 0) if st == 200 else 0
    if not 0 < ctx <= 400_000:
        print(f"  note: context {ctx}: not checked")
        return
    text = "word " * (ctx + 64)
    msgs = [{"role": "user", "content": text}]
    st, r = cl.post("/v1/chat/completions", {"model": model, "messages": msgs, "max_tokens": 8})
    err = (r or {}).get("error") or {}
    check(st == 400 and err.get("code") == "context_length_exceeded" and
          "maximum context length" in err.get("message", ""), "chat completions: 400 context_length_exceeded",
          f"{st} {json.dumps(r)[:200]}")
    evs = list(cl.stream("/v1/chat/completions", {"model": model, "messages": msgs, "max_tokens": 8, "stream": True}))
    check(bool(evs) and evs[0][0] == "http_error" and evs[0][1]["status"] == 400 and
          "context_length_exceeded" in evs[0][1]["body"], "chat completions, streamed: 400 before the stream",
          str(evs[:1])[:200])
    st, r = cl.post("/v1/responses", {"model": model, "input": text, "max_output_tokens": 16})
    check(st == 400 and ((r or {}).get("error") or {}).get("code") == "context_length_exceeded",
          "responses: 400 context_length_exceeded", f"{st} {json.dumps(r)[:200]}")
    st, r = cl.post("/v1/messages", {"model": model, "max_tokens": 8, "messages": msgs})
    err = (r or {}).get("error") or {}
    check(st == 400 and err.get("type") == "invalid_request_error" and
          err.get("message", "").startswith("prompt is too long"), "messages: 400 prompt is too long",
          f"{st} {json.dumps(r)[:200]}")
    evs = list(cl.stream("/v1/messages", {"model": model, "max_tokens": 8, "messages": msgs, "stream": True}))
    check(bool(evs) and evs[0][0] == "http_error" and evs[0][1]["status"] == 400 and
          "prompt is too long" in evs[0][1]["body"], "messages, streamed: 400 before the stream", str(evs[:1])[:200])
    st, r = cl.post("/v1/chat/completions", {
        "model": model, "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False}})
    check(st == 200 and r["usage"]["completion_tokens"] > 0, "and the next request is served", json.dumps(r)[:200])


def security_checks(cl: Client, model: str):
    """Input depth limits, malformed JSON types, dashboard path checks, and loopback browser guards."""
    print("Security checks")
    st, r = cl.post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hi"}],
                                             "tool_choice": 123, "max_tokens": 4})
    check(st == 400, "wrong JSON type for tool_choice -> 400 (not 500)", f"{st} {str(r)[:200]}")
    st, r = cl.post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "hi"}],
                                             "response_format": {"type": 123}, "max_tokens": 4})
    check(st == 400, "wrong JSON type for response_format.type -> 400", f"{st} {str(r)[:200]}")
    # An object, as a request is: a bare array is refused for not being one before its depth is looked at.
    deep = '{"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"metadata":{"a":' + "[" * 80 + "]" * 80 + "}}"
    c = cl._conn()
    try:
        c.request("POST", "/v1/chat/completions", body=deep, headers=cl._headers())
        resp = c.getresponse()
        said = resp.read()
        check(resp.status == 400 and b"nested more than" in said, "deeply nested JSON body -> 400, for its depth",
              f"{resp.status} {said[:120]}")
    finally:
        c.close()
    st, _, _ = cl.get_raw("/dashboard/C:/Windows/win.ini")
    check(st == 404, "dashboard rejects Windows drive-letter path", str(st))
    if not cl.key:
        st, _, _ = cl.get_raw("/v1/requests", headers={"Origin": "https://evil.example"})
        check(st == 403, "cross-origin Origin header rejected on /v1/requests", str(st))
        st, _, _ = cl.get_raw("/v1/requests", headers={"Host": "evil.example:8080"})
        check(st == 403, "non-loopback Host header rejected (DNS rebinding guard)", str(st))
        st, _, _ = cl.get_raw("/v1/requests", headers={"Sec-Fetch-Site": "cross-site"})
        check(st == 403, "Sec-Fetch-Site: cross-site rejected", str(st))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--key")
    a = ap.parse_args()
    cl = Client(a.url, a.key)
    st, h = cl.get("/health")
    check(st == 200 and h["status"] == "ok", "health")
    st, m = cl.get("/v1/models")
    model = m["data"][0]["id"] if st == 200 else "model"
    check(st == 200, f"models: {model}")
    streaming_flag_checks(cl, model)
    openai_checks(cl, model)
    tool_choice_checks(cl, model)
    responses_checks(cl, model)
    anthropic_checks(cl, model)
    context_checks(cl, model)
    concurrent_checks(cl, model)
    superseded_response_checks(cl, model)
    nonstream_disconnect_checks(cl, model)
    telemetry_checks(cl, model)
    dashboard_checks(cl)
    model_panel_checks(cl, model)
    request_log_checks(cl, model)
    failed_switch_checks(cl, model)
    system_log_checks(cl)
    security_checks(cl, model)
    unload_checks(cl, model)  # last: the model goes and comes back
    print("ALL OK" if FAILS == 0 else f"{FAILS} CHECKS FAILED")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
