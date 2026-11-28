# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Compare CLI rendering, the actual local prompts and all four HTTP chat routes."""
import json
import math
import subprocess

from local_server import Client, serve, server_env

MESSAGES = [{"role": "user", "content": "hello"}]
DEFAULTS = {"enable_thinking": False, "preserve_thinking": True, "reasoning_effort": "medium",
            "messages": [], "tools": ["fake"], "add_generation_prompt": False}
TEMPLATE = ("{{ messages[0].content }}{% if tools %} wrong tools{% endif %}"
            "{% if add_generation_prompt %} generate{% endif %}"
            "{% if preserve_thinking %} preserve{% endif %} {{ reasoning_effort }}"
            "{% if enable_thinking %} thinking enabled{% endif %}")


def check_template_kwargs(exe, model, root):
    root.mkdir()
    env = server_env(root, exe)
    source = root / "chat.jinja"
    source.write_text(TEMPLATE, encoding="utf-8")
    history = root / "messages.json"
    history.write_text(json.dumps(MESSAGES), encoding="utf-8")
    flags = ["--chat-template", source, "--chat-template-kwargs", json.dumps(DEFAULTS)]

    def local(command, *args, stdin=None):
        return subprocess.run([exe, command, model, "--backend", "cpu_ref", *args], input=stdin,
                              capture_output=True, encoding="utf-8", env=env, check=True, timeout=30)

    def rendered(options, *extra):
        r = local("template", history, "--chat-template", source, *extra, "--chat-template-kwargs", json.dumps(options))
        return json.loads(r.stdout)

    base = rendered(DEFAULTS)
    assert base["text"] == "hello generate preserve medium", base
    on = dict(DEFAULTS, enable_thinking=True, reasoning_effort="high")
    explicit = rendered(on)
    assert explicit["text"] == "hello generate preserve high thinking enabled", explicit
    assert rendered(dict(DEFAULTS, enable_thinking=True), "--no-think") == base
    expected = len(base["ids"])
    overridden = len(explicit["ids"])
    assert expected != overridden
    for run in (local("run", "--chat", "-p", "hello", "-n", "1", *flags),
                local("chat", "-n", "1", *flags, stdin="hello\n\n")):
        assert f"[prompt {expected} " in run.stderr, run.stderr

    def check_routes(url):
        client = Client(url, None)
        anthropic = {"thinking": {"type": "enabled"}, "output_config": {"effort": "high"}}
        routes = [
            ("chat/completions", {"messages": MESSAGES, "max_tokens": 1}, {"reasoning_effort": "high"}),
            ("responses", {"input": "hello", "max_output_tokens": 1}, {"reasoning": {"effort": "high"}}),
            ("messages", {"messages": MESSAGES, "max_tokens": 1}, anthropic),
            ("messages/count_tokens", {"messages": MESSAGES}, anthropic),
        ]
        for route, request, structured in routes:
            nested = {"enable_thinking": False, "reasoning_effort": "medium"}
            for fields, count, temp in (({}, expected, 0.7), (structured, overridden, 0.6),
                                        (dict(structured, chat_template_kwargs=nested), expected, 0.7)):
                status, body = client.post("/v1/" + route, dict(model="test model", **request, **fields))
                assert status == 200, (route, fields, status, body)
                usage = body.get("usage", body)
                tokens = usage.get("input_tokens", usage.get("prompt_tokens")) + usage.get("cache_read_input_tokens", 0)
                assert tokens == count, (route, fields, count, usage)
                if route == "responses":
                    assert math.isclose(body["temperature"], temp, rel_tol=1e-6), (fields, body["temperature"])
        # An explicit request for thinking normally conflicts with prefill-only Anthropic Messages.
        for enable, status in ((True, 400), (False, 200)):
            got, body = client.post("/v1/messages", {"model": "test model", "messages": MESSAGES, "max_tokens": 0,
                                                     "thinking": {"type": "enabled"},
                                                     "chat_template_kwargs": {"enable_thinking": enable}})
            assert got == status, (enable, got, body)

    serve(exe, root, env, [model], check_routes, options=["--name", "test model", *flags])
