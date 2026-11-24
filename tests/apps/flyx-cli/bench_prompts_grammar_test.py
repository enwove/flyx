# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""flyx bench --prompts where a chat without thinking carries a reply grammar: the tiny Inkling-2 model (float64
backend), whose format can keep thinking out of a reply only with one.  The set's prompts are all prepared before the
first runs, so each request has to keep its grammar until its turn: every reply then opens the text section the
grammar opens, written with the format's control tokens or spelled out in bytes, whichever the model's scores prefer.

    python tests/apps/flyx-cli/bench_prompts_grammar_test.py FLYX TINY_INKLING2.gguf   (inkling2_cpu_ref writes it)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

FLYX, MODEL = sys.argv[1:3]
OPENER = "<|message_model|><|content_text|>"
failures = []


def check(ok, what, got=None):
    print(f"  {'ok' if ok else 'FAIL':4s} {what}" + ("" if ok or got is None else f": got {got!r}"))
    if not ok:
        failures.append(what)


def flyx(*args):
    return subprocess.run([FLYX, *args], capture_output=True, text=True, encoding="utf8")


markers = flyx("tokenize", MODEL, OPENER).stdout.split()
check(len(markers) == 2, "the opener is two control tokens of the model", markers)
text_of = dict(zip((int(m) for m in markers), ("<|message_model|>", "<|content_text|>")))


def opening(ids):
    """The reply's text up to the first token that is neither a marker of the opener nor a byte (ids below 256)."""
    out = ""
    for t in ids:
        if t in text_of:
            out += text_of[t]
        elif t < 256:
            out += chr(t)
        else:
            break
    return out


with tempfile.TemporaryDirectory() as tmp:
    prompt_set = os.path.join(tmp, "set.jsonl")
    with open(prompt_set, "w", encoding="utf8") as f:
        f.write(json.dumps({"question": "hello"}) + "\n")
        f.write(json.dumps({"messages": [{"role": "user", "content": "in"}]}) + "\n")
        f.write(json.dumps({"question": "hello hello"}) + "\n")
    # 40 tokens: the opener spelled a byte a token is 33
    r = flyx("bench", MODEL, "--prompts", prompt_set, "--backend", "cpu_ref", "--no-spec", "--no-lookup", "-n", "40",
             "--ignore-eos", "--json")
    try:
        rec = json.loads(r.stdout)
    except ValueError:
        rec = None
    check(r.returncode == 0 and rec is not None and len(rec["prompts"]) == 3,
          "three chats of a prompt set run, each under its own reply grammar", (r.returncode, r.stderr[-400:]))
    for p in rec["prompts"] if rec else []:
        check(opening(p["ids"]).startswith(OPENER), f"prompt {p['i']}: the reply opens a text section",
              (p["ids"][:8], opening(p["ids"])[:40]))

print(f"{len(failures)} failed" if failures else "all passed")
sys.exit(1 if failures else 0)
