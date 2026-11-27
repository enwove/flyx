# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""flyx bench --prompts where a chat without thinking carries a reply grammar: the tiny Inkling-2 model (float64
backend), whose format can keep thinking out of a reply only with one.  The set's prompts are all prepared before the
first runs, so each request has to keep its grammar until its turn. Every reply must open its text section with
the format's two control tokens.

    python tests/apps/flyx-cli/bench_prompts_grammar_test.py FLYX TINY_INKLING2.gguf   (inkling2_cpu_ref writes it)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from check import check, finish  # noqa: E402

FLYX, MODEL = sys.argv[1:3]
OPENER = "<|message_model|><|content_text|>"


def flyx(*args):
    return subprocess.run([FLYX, *args], capture_output=True, text=True, encoding="utf8")


markers = [int(token) for token in flyx("tokenize", MODEL, OPENER).stdout.split()]
check(len(markers) == 2, "the opener is two control tokens of the model", markers)


with tempfile.TemporaryDirectory() as tmp:
    prompt_set = os.path.join(tmp, "set.jsonl")
    with open(prompt_set, "w", encoding="utf8") as f:
        f.write(json.dumps({"question": "hello"}) + "\n")
        f.write(json.dumps({"messages": [{"role": "user", "content": "in"}]}) + "\n")
        f.write(json.dumps({"question": "hello hello"}) + "\n")
    r = flyx("bench", MODEL, "--prompts", prompt_set, "--backend", "cpu_ref", "--no-spec", "--no-lookup", "-n", "2",
             "--ignore-eos", "--json")
    try:
        rec = json.loads(r.stdout)
    except ValueError:
        rec = None
    check(r.returncode == 0 and rec is not None and len(rec["prompts"]) == 3,
          "three chats of a prompt set run, each under its own reply grammar", (r.returncode, r.stderr[-400:]))
    for p in rec["prompts"] if rec else []:
        check(p["ids"] == markers, f"prompt {p['i']}: the reply opens a text section with its control tokens", p["ids"])

sys.exit(finish("bench_prompts_grammar_test"))
