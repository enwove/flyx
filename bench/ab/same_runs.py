# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Two `flyx bench --prompts --json` records of one command, from two builds: the same answers and the same rounds.

    python -I bench/ab/same_runs.py NEW.json OLD.json [--answers-only]

A change that should change no output (and, with a drafter, no round) is held to a build before it: over the
prompts both records hold (by their index in the prompt set: a --limit run against a whole one compares its first
N), every prompt's generated tokens must be the older record's and, unless --answers-only, its rounds too, round by
round (each round's tokens and drafts accepted; a record made before `trace` had a third field compares on these
two).  --answers-only: the answers alone, also across drafters, widths, lookup and backend options (speculation must
not change an answer: a drafter's record against the plain one's).  Fails (exit 1) when they differ, when the two ran
another command (model file, prompt set, suffix, K/V, context, sampling, drafter, lookup, backend options, end tokens,
token limit, the rounds offered), when either run failed (its record's `error`), when a prompt's tokens differ, or when
no prompt is shared.  Prints what it compared and each record's decode tok/s over those
prompts, as its own run measured them (two builds' speed is an A/B taking turns: bench/ab/ab.sh).
"""
from __future__ import annotations

import argparse
import json
import sys

# What the two runs must share (the record's fields; compared where both records have one).
SAME = ("model", "prompt_set", "suffix", "kv", "ctx", "sampling", "drafter", "lookup", "backend_options", "eos_ids",
        "max_new_tokens", "ignore_eos", "rounds")
# How the answers were drafted, which --answers-only compares across.
DRAFTING = ("drafter", "lookup", "backend_options", "rounds")


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rounds_of(prompt):
    """A prompt's rounds as (tokens, drafts accepted): the fields every record's trace has."""
    return [tuple(step[:2]) for step in prompt.get("trace", [])]


def first_difference(a, b):
    """Where two lists part (the shorter one's length when one starts the other)."""
    return next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def compare(new, old, answers_only=False):
    """(failures, lines): what differs, and what was compared."""
    fails = []
    for side, rec in (("new", new), ("old", old)):
        if "error" in rec:  # a run that stopped at a failed prompt (flyx bench records the prompts before it)
            fails.append(f"the {side} run failed: {rec['error']}")
    for key in SAME:
        if answers_only and key in DRAFTING:
            continue
        if key in new and key in old and new[key] != old[key]:
            fails.append(f"the runs differ in {key}: {new[key]!r} against {old[key]!r}")
    by_index = {p["i"]: p for p in old.get("prompts", [])}
    shared = [p for p in new.get("prompts", []) if p["i"] in by_index]
    if not shared:
        fails.append("no prompt is in both records")
    answers = rounds = n_rounds = tokens = 0
    toks = {"new": 0, "old": 0}
    ms = {"new": 0.0, "old": 0.0}
    for p in shared:
        q = by_index[p["i"]]
        if p.get("prompt_ids") != q.get("prompt_ids"):
            fails.append(f"prompt {p['i']}: other prompt tokens")
            continue
        if p["ids"] == q["ids"]:
            answers += 1
        else:
            fails.append(f"prompt {p['i']}: another answer from token {first_difference(p['ids'], q['ids'])} "
                         f"({len(p['ids'])} tokens against {len(q['ids'])})")
        r, s = rounds_of(p), rounds_of(q)
        if r == s:
            rounds += 1
        elif not answers_only:
            fails.append(f"prompt {p['i']}: other rounds from round {first_difference(r, s)} ({len(r)} rounds "
                         f"against {len(s)})")
        n_rounds += len(r)
        tokens += len(p["ids"])
        for side, rec in (("new", p), ("old", q)):
            toks[side] += max(0, rec.get("new_tokens", 0) - 1)  # after the first, as flyx bench counts decode
            ms[side] += rec.get("decode_ms", 0.0)
    n = len(shared)
    rate = {side: (toks[side] / ms[side] * 1000 if ms[side] > 0 else 0.0) for side in toks}
    lines = [f"{n} prompts compared (of {len(new.get('prompts', []))} and {len(old.get('prompts', []))}), "
             f"{tokens} tokens, {n_rounds} rounds: answers the same {answers} of {n}, rounds the same {rounds} of {n}"
             + (" (not required)" if answers_only else ""),
             f"decode tok/s over these prompts, as each run measured it: new {rate['new']:.1f}, old {rate['old']:.1f}"]
    return fails, lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("new", help="the record of the build under test")
    ap.add_argument("old", help="the record of the build before it, the same command")
    ap.add_argument("--answers-only", action="store_true",
                    help="the answers alone must be the same (not the rounds), whatever drafted them")
    args = ap.parse_args(argv)
    fails, lines = compare(read(args.new), read(args.old), args.answers_only)
    for line in lines:
        print(line)
    for f in fails[:20]:
        print("FAIL: " + f)
    if len(fails) > 20:
        print(f"FAIL: ... {len(fails) - 20} more")
    print("FAIL" if fails else "PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
