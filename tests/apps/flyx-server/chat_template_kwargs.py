# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Compare CLI rendering, the actual local prompts and all four HTTP chat routes."""
import json
import math
from pathlib import Path
import subprocess
import sys

from local_server import Client, serve, server_env, wait_load

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python" / "convert"))
from flyx_convert.fileformat import Reader, Writer

MESSAGES = [{"role": "user", "content": "hello"}]
DEFAULTS = {"enable_thinking": False, "preserve_thinking": True, "reasoning_effort": "medium",
            "messages": [], "tools": ["fake"], "add_generation_prompt": False}
TEMPLATE = ("{{ messages[0].content }}{% if tools %} wrong tools{% endif %}"
            "{% if add_generation_prompt %} generate{% endif %}"
            "{% if preserve_thinking %} preserve{% endif %}"
            "{% if reasoning_effort is defined %} {{ reasoning_effort }}{% endif %}"
            "{% if enable_thinking %} thinking enabled{% endif %}")


def input_tokens(body):
    usage = body.get("usage", body)
    return usage.get("input_tokens", usage.get("prompt_tokens")) + usage.get("cache_read_input_tokens", 0)


def check_inkling_effort(exe, model, root):
    """The tiny Inkling-2 GGUF's own template writes '0' when the format supplies the effort 'none'."""
    env = server_env(root, exe)
    history = root / "messages.json"
    history.write_text(json.dumps(MESSAGES), encoding="utf-8")
    medium = ["--chat-template-kwargs", '{"reasoning_effort":"medium"}']
    none = ["--chat-template-kwargs", '{"reasoning_effort":"none"}']

    def local(cmd, *args, stdin=None):
        return subprocess.run([exe, cmd, model, "--backend", "cpu_ref", *args], input=stdin,
                              capture_output=True, encoding="utf-8", env=env, check=True, timeout=30)

    def prompt(effort, *args):
        rendered = json.loads(local("template", history, *args).stdout)
        assert f"Thinking effort level: {effort}<|end_message|>" in rendered["text"], rendered
        return len(rendered["ids"])

    off = prompt("0", "--no-think", *medium)
    assert prompt("0", *none) == off
    on = prompt("medium", *medium)
    native = prompt("0.9")
    assert len({on, off, native}) == 3
    for flags in (none, [*medium, "--no-think"]):
        run = local("run", "--chat", "-p", "hello", "-n", "2", *flags)
        assert run.stdout == "<|message_model|><|content_text|>", run.stdout  # no-thinking grammar
        chat = local("chat", "-n", "2", *flags, stdin="hello\n\n")
        assert all(f"[prompt {off} " in r.stderr for r in (run, chat)), (run.stderr, chat.stderr)

    def check_routes(url, default_count, reenabled_count, default_thinking):
        client = Client(url, None)
        # The default a request without enable_thinking gets, as the dashboard's Chat shows it.
        status, tm = client.get("/v1/telemetry")
        assert status == 200 and tm["server"]["enable_thinking"] is default_thinking, (status, tm.get("server"))

        def post(route, token_count, **fields):
            body = {"input": "hello", "max_output_tokens": 2} if route == "responses" else {"messages": MESSAGES}
            if route in ("chat/completions", "messages"):
                body["max_tokens"] = 2
            status, result = client.post("/v1/" + route, dict(body, model="test model", **fields))
            assert status == 200 and input_tokens(result) == token_count, (route, fields, status, result)
            if route == "responses":
                assert math.isclose(result["temperature"], 0.7 if token_count == off else 0.6, rel_tol=1e-6), result

        for route in ("chat/completions", "responses", "messages", "messages/count_tokens"):
            post(route, default_count)
            post(route, off, chat_template_kwargs={"enable_thinking": False})
        post("chat/completions", off, enable_thinking=False)
        post("chat/completions", reenabled_count, enable_thinking=True)
        post("responses", on, reasoning={"effort": "medium"})
        for route in ("messages", "messages/count_tokens"):
            post(route, off, thinking={"type": "disabled"})

    # Telemetry's default as a request without a mode is prompted: on for an effort, off for the effort "none".
    for flags, default_count, reenabled, thinking in ((medium, on, on, True), (none, off, native, False),
                                                      ([*medium, "--no-think"], off, on, False)):
        serve(exe, root, env, [model], lambda url: check_routes(url, default_count, reenabled, thinking),
              options=["--name", "test model", *flags])


def check_template_kwargs(exe, model, root):
    root.mkdir()
    original = model
    model = root / "recommended.flyx"
    with Reader(original) as reader:
        writer = Writer()
        writer.meta = dict(reader.header, recommended={"sampling": {
            "thinking": {"temperature": 0.6}, "non_thinking": {"temperature": 0.7},
            "thinking_coding": {"temperature": 0.25}, "default": {"top_k": 6, "min_p": 0.1}}})
        for name, rec in reader.header["tensors"].items():
            writer.add_tensor(name, rec["dtype"], rec["shape"], bytes(reader.raw(rec)))
        for name in reader.header["blobs"]:
            writer.add_blob(name, reader.blob(name))
        writer.write(model)
    env = server_env(root, exe)
    source = root / "chat.jinja"
    source.write_text(TEMPLATE, encoding="utf-8")
    history = root / "messages.json"
    history.write_text(json.dumps(MESSAGES), encoding="utf-8")
    flags = ["--chat-template", source, "--chat-template-kwargs", json.dumps(DEFAULTS)]

    def local(command, *args, stdin=None):
        return subprocess.run([exe, command, model, "--backend", "cpu_ref", *args], input=stdin,
                              capture_output=True, encoding="utf-8", errors="replace", env=env, check=True, timeout=30)

    def rendered(options, *extra):
        r = local("template", history, "--chat-template", source, *extra, "--chat-template-kwargs", json.dumps(options))
        return json.loads(r.stdout)

    chat_flags = ["--seed", "17", "-n", "8", *flags]
    inherited = local("chat", *chat_flags, stdin="hello\n\n")
    explicit = local("chat", *chat_flags, "--temperature", "0.7", "--top-k", "6", stdin="hello\n\n")
    assert inherited.stdout == explicit.stdout
    load_log = next(line for line in inherited.stderr.splitlines() if line.startswith("sampling_defaults "))
    assert json.loads(load_log.split(" ", 1)[1])["non_thinking"]["top_k"] == {"value": 6, "origin": "model_recommended"}
    for command in ("chat", "serve"):
        bad = subprocess.run([exe, command, model, "--sampling-profile", "unknown"], env=env,
                             capture_output=True, encoding="utf-8", timeout=30)
        assert bad.returncode == 2 and "--sampling-profile expects" in bad.stderr, bad
    coding = local("chat", *chat_flags, "--sampling-profile", "thinking_coding", stdin="hello\n\n")
    log = next(line for line in coding.stderr.splitlines() if line.startswith("sampling_defaults "))
    assert json.loads(log.split(" ", 1)[1])["thinking"]["temperature"]["value"] == 0.25
    base = rendered(DEFAULTS)
    assert base["text"] == "hello generate preserve", base
    on = dict(DEFAULTS, enable_thinking=True, reasoning_effort="high")
    explicit = rendered(on)
    assert explicit["text"] == "hello generate preserve high thinking enabled", explicit
    assert rendered(dict(DEFAULTS, enable_thinking=True), "--no-think") == base
    expected = len(base["ids"])
    overridden = len(explicit["ids"])
    explicit_off = len(local("tokenize", "hello generate preserve medium").stdout.split())
    assert len({expected, overridden, explicit_off}) == 3
    for run in (local("run", "--chat", "-p", "hello", "-n", "1", *flags),
                local("chat", "-n", "1", *flags, stdin="hello\n\n")):
        assert f"[prompt {expected} " in run.stderr, run.stderr

    def check_routes(url):
        client = Client(url, None)
        status, tm = client.get("/v1/telemetry")
        assert status == 200 and tm["server"]["enable_thinking"] is False, (status, tm.get("server"))
        defaults = tm["sampling_defaults"]
        assert defaults["non_thinking"]["temperature"]["origin"] == "model_recommended"
        assert defaults["thinking_profile"] == "thinking_coding"
        assert defaults["thinking"]["temperature"] == {"value": 0.25, "origin": "model_recommended"}
        assert defaults["thinking"]["top_k"] == {"value": 0, "origin": "cli"}
        assert defaults["thinking"]["frequency_penalty"] == {"value": 0, "origin": "fallback"}
        assert defaults["warnings"] == ["recommended.sampling.default.min_p: unsupported, ignored"]
        anthropic = {"thinking": {"type": "enabled"}, "output_config": {"effort": "high"}}
        routes = [
            ("chat/completions", {"messages": MESSAGES, "max_tokens": 1}, {"reasoning_effort": "high"}),
            ("responses", {"input": "hello", "max_output_tokens": 1}, {"reasoning": {"effort": "high"}}),
            ("messages", {"messages": MESSAGES, "max_tokens": 1}, anthropic),
            ("messages/count_tokens", {"messages": MESSAGES}, anthropic),
        ]
        for route, request, structured in routes:
            nested = {"enable_thinking": False, "reasoning_effort": "medium"}
            for fields, count, temp in (({}, expected, 0.7), (structured, overridden, 0.25),
                                        (dict(structured, chat_template_kwargs=nested), explicit_off, 0.7)):
                status, body = client.post("/v1/" + route, dict(model="test model", **request, **fields))
                assert status == 200, (route, fields, status, body)
                assert input_tokens(body) == count, (route, fields, count, body)
                if route == "responses":
                    assert math.isclose(body["temperature"], temp, rel_tol=1e-6), (fields, body["temperature"])
        # An explicit request for thinking normally conflicts with prefill-only Anthropic Messages.
        for enable, status in ((True, 400), (False, 200)):
            got, body = client.post("/v1/messages", {"model": "test model", "messages": MESSAGES, "max_tokens": 0,
                                                     "thinking": {"type": "enabled"},
                                                     "chat_template_kwargs": {"enable_thinking": enable}})
            assert got == status, (enable, got, body)

        status, body = client.post("/v1/responses", {"input": "hello", "max_output_tokens": 1,
                                                      "chat_template_kwargs": {"enable_thinking": True},
                                                      "sampling_profile": "thinking_coding"})
        assert status == 200 and body["temperature"] == 0.25, body
        status, body = client.post("/v1/responses", {"input": "hello", "max_output_tokens": 1, "temperature": 0})
        assert status == 200 and body["temperature"] == 0, body
        status, body = client.post("/v1/load", {"model": str(original), "name": "second", "ctx": 128})
        assert status in (200, 202) and wait_load(client, 30)["state"] == "ready", body
        _, tm = client.get("/v1/telemetry")
        assert tm["model"]["id"] == "second" and tm["sampling_defaults"]["warnings"] == []
        field = tm["sampling_defaults"]["thinking"]["temperature"]
        assert field["origin"] == "fallback" and math.isclose(field["value"], 0.6, rel_tol=1e-6)
        status, body = client.post("/v1/responses", {"input": "hello", "max_output_tokens": 1})
        assert status == 200 and math.isclose(body["temperature"], 0.7, rel_tol=1e-6), body

    serve(exe, root, env, [model], check_routes, options=["--name", "test model", "--sampling-profile", "thinking_coding", "--top-k", "0", *flags])
