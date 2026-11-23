# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Read prompts, templates and messages and write the decode trace through UTF-8 CLI paths."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


exe, model = (Path(arg).resolve() for arg in sys.argv[1:])
with tempfile.TemporaryDirectory(prefix="flyx-paths-") as tmp:
    work = Path(tmp)
    directory = work / "测试 café"
    directory.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("FLYX_")}

    def run(*args):
        return subprocess.run([exe, *args], env=env, cwd=work, check=True, stdout=subprocess.PIPE,
                              timeout=45).stdout

    prompt = directory / "输入.txt"
    prompt.write_text("hello", encoding="utf-8")
    trace = directory / "生成.jsonl"
    flags = ["--backend", "cpu_ref", "--ctx", "128", "-n", "3", "--ignore-eos"]
    assert run("run", model, "-f", prompt, "--rounds", trace, *flags) == run("run", model, "-p", "hello", *flags)
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    assert rows and all("tokens" in row for row in rows), rows

    template = directory / "模板.jinja"
    template.write_text("{% for m in messages %}{{ m.content }}{% endfor %}", encoding="utf-8")
    messages = directory / "对话.json"
    messages.write_text(json.dumps([{"role": "user", "content": "hello"}]), encoding="utf-8")
    assert json.loads(run("template", model, messages, "--chat-template", template))["text"] == "hello"
