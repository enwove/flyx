# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""flyx bench --prompts on the tiny model (float64 backend): each line's prompt as the template renders it (a question
with --suffix the same as its message written out; token ids as given), every prompt's generated tokens and rounds in
the record (a round whose last token, the end token, was a draft marked so), the drafted run's answers those of plain
decoding (tests/e2e/check_dv.py, which refuses a drafted run as plain decoding), the record on stdout with --json
alone, a bad line refused by its number before any prompt runs, and an output file that cannot be written before the
model loads; the built-in prompts' --json as before; --cycle's rounds past the engine's refused at load.

    python tests/apps/flyx-cli/bench_prompts_test.py FLYX ASSETS TEMPLATE   (TEMPLATE: tests/core/render/templates/
                                                                            qwen3.5.jinja, the tiny model's has none)
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "e2e"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))

import check_dv  # noqa: E402
from check import check, finish  # noqa: E402

FLYX, ASSETS, TEMPLATE = sys.argv[1:4]
MODEL = os.path.join(ASSETS, "tiny.flyx")  # MTP drafts and lookup rounds: rounds that accept drafts
BS = chr(92)
QUESTION = "What is 2 + 3?"
SUFFIX_ARG = BS + "nAnswer in " + BS + "boxed{}."  # as typed: \n a newline, \b kept
SUFFIX = "\nAnswer in " + BS + "boxed{}."
# A prompt whose answer, on the tiny model, is a stretch the prompt holds, then 0, an end token: a lookup round drafts
# all of it after the first token, the end token last, and accepts it (tiny.flyx's own answer, found by a search;
# a new tiny model needs another).
STOP_PROMPT = [438, 358, 741, 368, 506, 355, 204, 327, 349, 135, 299, 603, 211, 307, 94]
STOP_ANSWER = [319, 319, 106, 106, 106, 106, 106] + [407] * 8 + [0]


def flyx(*args):
    return subprocess.run([FLYX, *args], capture_output=True, text=True, encoding="utf8")


def check_dv_quietly(argv):
    with contextlib.redirect_stdout(io.StringIO()):
        return check_dv.main(argv)


with tempfile.TemporaryDirectory() as tmp:
    prompt_set = os.path.join(tmp, "set.jsonl")
    with open(prompt_set, "w", encoding="utf-8-sig") as f:  # a byte order mark, as PowerShell's utf8 writes one
        f.write(json.dumps({"question": QUESTION, "answer": "#### 5"}) + "\n\n")
        f.write(json.dumps({"messages": [{"role": "user", "content": QUESTION + SUFFIX}]}) + "\r\n")
        f.write(json.dumps({"ids": [1, 5, 6, 7]}) + "\n")
        f.write(json.dumps({"ids": STOP_PROMPT + STOP_ANSWER + STOP_PROMPT}) + "\n")
    # 64 tokens: the tiny model's answers come to repeat themselves, and lookup rounds accept their drafts
    common = ["bench", MODEL, "--prompts", prompt_set, "--suffix", SUFFIX_ARG, "--backend", "cpu_ref",
              "--chat-template", TEMPLATE, "-n", "64"]
    drafted, plain = os.path.join(tmp, "drafted.json"), os.path.join(tmp, "plain.json")
    r = flyx(*common, "--json", drafted)
    check(r.returncode == 0, "the drafted run", r.stderr)
    r = flyx(*common, "--no-spec", "--no-lookup", "--json", plain)
    check(r.returncode == 0, "the plain run", r.stderr)
    check("4 prompts" in r.stdout, "the totals on stdout when the record goes to a file", r.stdout)
    rec = json.load(open(drafted, encoding="utf8"))
    ps = rec["prompts"]

    messages = os.path.join(tmp, "messages.json")
    with open(messages, "w", encoding="utf8") as f:
        json.dump([{"role": "user", "content": QUESTION + SUFFIX}], f)
    rendered = json.loads(flyx("template", MODEL, messages, "--no-think", "--chat-template", TEMPLATE).stdout)["ids"]
    check([p["i"] for p in ps] == [0, 1, 2, 3], "prompts by their place in the set, blank lines skipped")
    check(rec["suffix"] == SUFFIX, "the suffix: \\n a newline, the other backslash itself", rec["suffix"])
    check(ps[0]["prompt_ids"] == rendered, "a question: rendered with its suffix, thinking off (flyx template)")
    check(ps[1]["prompt_ids"] == rendered, "messages: rendered as written")
    check(ps[2]["prompt_ids"] == [1, 5, 6, 7], "token ids: the prompt as given")
    check(rec["max_new_tokens"] == 64 and rec["kv"] in ("16", "fp8") and rec["eos_ids"], "what the run used")
    accepted = 0
    for p in ps:
        stopped = p["finish"] == "stop"
        check(len(p["trace"]) == p["rounds"] and all(len(step) == 4 and -1 <= step[3] < len(rec["rounds"])
                                                     for step in p["trace"]),
              f"prompt {p['i']}: [tokens, accepted, last a draft, the round it was (-1 lookup)] a round")
        check(p["new_tokens"] == len(p["ids"]) - stopped and p["new_tokens"] <= 64,
              f"prompt {p['i']}: the generated tokens, the end token after them when it ended")
        check(1 + sum(n for n, _, _, _ in p["trace"]) >= len(p["ids"])
              and all(n == a + 1 and not last for n, a, last, _ in p["trace"][:-1]),
              f"prompt {p['i']}: the rounds gave those tokens, each its accepted drafts and one more, a draft last "
              "only in the last")
        accepted += sum(a for _, a, _, _ in p["trace"])
    check(accepted > 0, "the drafted run accepted drafts")
    check(ps[3]["ids"] == STOP_ANSWER and [step[:3] for step in ps[3]["trace"]] == [[15, 14, 1]],
          "the end token accepted as a draft: the round's 15 tokens, 14 accepted drafts, its last token a draft",
          (ps[3]["ids"], ps[3]["trace"]))
    rows = check_dv.per_prompt(rec)
    check(rows[3]["len"] == 16 and rows[3]["need"] == 1,
          "check_dv: one forward commits all 16 tokens (bifill-dv counts the end token among the drafts)", rows[3])
    check(check_dv.per_prompt(json.load(open(plain, encoding="utf8")))[3]["need"] == 16, "check_dv: plain needs 16")
    check(check_dv_quietly(["--drafted", drafted, "--plain", plain]) == 0, "check_dv: the drafted answers are plain's")
    check(check_dv_quietly(["--drafted", drafted, "--plain", drafted]) == 1,
          "check_dv: a drafted record given as plain decoding fails")

    r = flyx("bench", MODEL, "--prompts", prompt_set, "--limit", "1", "--backend", "cpu_ref", "--json")
    try:
        one = json.loads(r.stdout)
    except ValueError:
        one = None
    check(one is not None and len(one["prompts"]) == 1, "--limit 1, --json alone: the record is all of stdout",
          r.stdout[:200])
    check(one is not None and one["max_new_tokens"] == 512, "a prompt set's answers get 512 tokens unless -n says")

    def refused(path, line, why, *extra):
        r = flyx("bench", MODEL, "--prompts", path, "--backend", "cpu_ref", "-n", "8", *extra)
        check(r.returncode != 0 and f"line {line}:" in r.stderr and " | prompt " not in r.stderr,
              f"{why}: refused by its line before any prompt runs", r.stderr)

    bad = os.path.join(tmp, "bad.jsonl")
    with open(bad, "w", encoding="utf8") as f:
        f.write(json.dumps({"question": QUESTION}) + "\n" + json.dumps({"prompt": QUESTION}) + "\n")
    refused(bad, 2, "a line with no prompt")
    vocab = os.path.join(tmp, "vocab.jsonl")
    with open(vocab, "w", encoding="utf8") as f:
        f.write(json.dumps({"ids": [1, 5]}) + "\n\n" + json.dumps({"ids": [1, 999999]}) + "\n")
    refused(vocab, 3, "a token outside the vocabulary")
    long = os.path.join(tmp, "long.jsonl")
    with open(long, "w", encoding="utf8") as f:
        f.write(json.dumps({"ids": [1, 5]}) + "\n" + json.dumps({"ids": [5] * 300}) + "\n")
    refused(long, 2, "a prompt longer than the context", "--ctx", "256")

    # --cycle names rounds by their index (-v lists them): an index past the rounds is refused at load, by the rounds
    # there are, before anything runs (it indexed past them on every reply); the rounds it names run, plain decoding's
    # text.
    r = flyx("run", MODEL, "--backend", "cpu_ref", "--cycle", "0,99", "-p", QUESTION, "-n", "4")
    check(r.returncode == 1 and "--cycle 99: the rounds are 0 (0), 1 (" in r.stderr,
          "--cycle past the rounds: refused at load, the rounds named", (r.returncode, r.stderr[-300:]))
    cycled = flyx("run", MODEL, "--backend", "cpu_ref", "--cycle", "0,1", "-p", QUESTION, "-n", "12")
    alone = flyx("run", MODEL, "--backend", "cpu_ref", "--no-spec", "--no-lookup", "-p", QUESTION, "-n", "12")
    check(cycled.returncode == 0 and cycled.stdout and cycled.stdout == alone.stdout,
          "--cycle 0,1: plain decoding's text", (cycled.returncode, cycled.stdout, alone.stdout, cycled.stderr[-300:]))

    nowhere = os.path.join(tmp, "no_such_dir", "out.json")
    for what, args in (("a prompt set", ["--prompts", prompt_set]), ("the built-in prompts", [])):
        r = flyx("bench", MODEL, *args, "--backend", "cpu_ref", "-n", "8", "--json", nowhere)
        check(r.returncode != 0 and "cannot write" in r.stderr and "prompt " not in r.stderr,
              f"{what}: an output file that cannot be written fails before anything runs", r.stderr)

    r = flyx("bench", MODEL, "--backend", "cpu_ref", "-n", "8", "--json")
    try:
        builtin = json.loads(r.stdout)
    except ValueError:
        builtin = None
    check(builtin is not None and len(builtin["prompts"]) == 3 and "decode_tok_s" in builtin,
          "the built-in prompts: --json still the record alone on stdout", r.stdout[:200])
    out = os.path.join(tmp, "builtin.json")
    r = flyx("bench", MODEL, "--backend", "cpu_ref", "-n", "8", "--json", out)
    check(r.returncode == 0 and os.path.exists(out) and "decode:" in r.stdout,
          "the built-in prompts: --json FILE, the summary on stdout", r.stdout)

sys.exit(finish("bench_prompts_test"))
