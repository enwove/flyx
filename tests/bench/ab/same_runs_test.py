# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""bench/ab/same_runs.py on records made up here: the same run passes, a --limit run against a whole one compares its
first prompts, a record from before `trace` had its third field compares on the first two; another answer, other
rounds (unless --answers-only, which also compares a drafter's record with the plain one's), a failed run, another
prompt's tokens, another command's field or no shared prompt fails.

    python tests/bench/ab/same_runs_test.py
"""
from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "bench", "ab"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))

import same_runs  # noqa: E402
from check import check, finish  # noqa: E402


def record(n, three_fields=True, kv="16"):
    prompts = []
    for i in range(n):
        ids = [10 + i, 11 + i, 12 + i, 2]
        trace = [[1, 0, 0], [3, 2, 1]] if three_fields else [[1, 0], [3, 2]]
        prompts.append({"i": i, "prompt_ids": [1, 5 + i], "ids": ids, "trace": trace, "new_tokens": len(ids),
                        "decode_ms": 4.0})
    return {"model": "m.flyx", "prompt_set": "p.jsonl", "suffix": "", "kv": kv, "ctx": 2048, "drafter": "dv",
            "lookup": False, "eos_ids": [2], "max_new_tokens": 512, "rounds": ["dv tree 15"], "prompts": prompts}


def run(new, old, *flags):
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for name, rec in (("new.json", new), ("old.json", old)):
            paths.append(os.path.join(d, name))
            with open(paths[-1], "w", encoding="utf-8") as f:
                json.dump(rec, f)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = same_runs.main(paths + list(flags))
        return code, out.getvalue()


code, out = run(record(3), record(3))
check(code == 0 and "answers the same 3 of 3, rounds the same 3 of 3" in out, "the same run passes")
code, out = run(record(2), record(5))
check(code == 0 and "2 prompts compared (of 2 and 5)" in out, "a --limit run against a whole one: its first prompts")
code, out = run(record(3), record(3, three_fields=False))
check(code == 0, "a trace of two fields against one of three: compared on the two")

plain = record(3)
plain.update(drafter="", rounds=[])
for p in plain["prompts"]:
    p["trace"] = [[1, 0, 0]] * len(p["ids"])
code, out = run(record(3), plain, "--answers-only")
check(code == 0 and "answers the same 3 of 3" in out, "--answers-only: a drafter's record against the plain one's")
code, out = run(record(3), plain)
check(code == 1 and "the runs differ in drafter" in out, "without --answers-only another drafter fails")

failed = record(1)
failed["error"] = "prompt 1: out of memory"
code, out = run(failed, record(3))
check(code == 1 and "the new run failed: prompt 1: out of memory" in out, "a run that failed fails, whatever matches")

other = record(3)
other["prompts"][1]["ids"][2] = 99
code, out = run(other, record(3))
check(code == 1 and "prompt 1: another answer from token 2" in out, "another answer fails, saying where")

other = record(3)
other["prompts"][2]["trace"][1] = [3, 1, 0]
code, out = run(other, record(3))
check(code == 1 and "prompt 2: other rounds from round 1" in out, "other rounds fail, saying where")
code, out = run(other, record(3), "--answers-only")
check(code == 0 and "rounds the same 2 of 3 (not required)" in out, "  but not with --answers-only")

other = record(3)
other["prompts"][0]["prompt_ids"] = [1, 6]
code, out = run(other, record(3))
check(code == 1 and "prompt 0: other prompt tokens" in out, "another prompt's tokens fail")

code, out = run(record(3, kv="fp8"), record(3))
check(code == 1 and "the runs differ in kv" in out, "another command's field fails")

shifted = copy.deepcopy(record(2))
for p in shifted["prompts"]:
    p["i"] += 10
code, out = run(shifted, record(2))
check(code == 1 and "no prompt is in both records" in out, "no shared prompt fails")

sys.exit(finish("same_runs_test"))
