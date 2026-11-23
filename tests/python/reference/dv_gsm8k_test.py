# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""python/reference/dv_gsm8k.py's results file (--out), on small stand-in files (nothing is loaded or run).

    python tests/python/reference/dv_gsm8k_test.py

* Every option that changes an answer or its count changes the settings the file keeps: the tree, the slots, the fill
  threshold, the length cap, the dtype, and the contents of the model directory, the questions, the canvas, the
  selector and the V-LoRA.  The options that only choose questions, check or compare do not.
* A run goes on after the questions a file holds only when the file was made with its settings.  A file made with
  other settings (the DYN rows offered to an ST run), or without a settings line, is refused, never mixed in; so is a
  run at another thread count than the rows' (float32 rounds differently).
* A row cut off by a stopped run is refused to a reader and dropped by a run that goes on; a byte order mark is read
  past; a line in the middle that is not JSON is refused.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python", "reference"))
import dv_gsm8k as runner  # noqa: E402

failures = 0


def check(ok, what):
    global failures
    print(("ok " if ok else "FAIL ") + what, flush=True)
    failures += not ok


def refused(path, settings):
    """The message open_results refuses `path` with, or None if it takes it."""
    try:
        _, f = runner.open_results(path, settings)
    except ValueError as e:
        return str(e)
    f.close()
    return None


def main():
    with tempfile.TemporaryDirectory() as d:
        def put(name, data):
            path = os.path.join(d, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
            return path
        model = os.path.dirname(put("model/config.json", b"{}"))
        put("model/model.safetensors", b"weights")
        put("model/README.md", b"not read")
        files = {n: put(n, n.encode()) for n in ("canvas.pt", "canvas2.pt", "sel.pt", "sel2.pt", "vlora.pt",
                                                 "questions.jsonl", "questions2.jsonl")}
        out = os.path.join(d, "results.jsonl")
        base = [model, "--canvas", files["canvas.pt"], "--selector", files["sel.pt"], "--questions",
                files["questions.jsonl"], "--out", out]

        def settings(*options):
            a = runner.arguments().parse_args(base + list(options))
            return runner.run_settings(a, runner.drafter(a, {"n": 32, "res_k": 16}, 99))
        s0 = settings()
        for option, value in (("--tree", "pool:31:7:1"), ("--tree", "beam:31:8"), ("--tree", "beam:15:8:cov"),
                              ("--slots", "64"), ("--fill", "0.5"), ("--max-new", "256"), ("--dtype", "float64"),
                              ("--canvas", files["canvas2.pt"]), ("--selector", files["sel2.pt"]),
                              ("--vlora", files["vlora.pt"]), ("--questions", files["questions2.jsonl"])):
            check(settings(option, value) != s0, f"{option} {os.path.basename(value)} changes the settings")
        put("model/model.safetensors", b"other weights")
        check(settings() != s0, "other weights in the model directory change the settings")
        put("model/model.safetensors", b"weights")
        put("model/README.md", b"read me")
        check(settings() == s0, "a README in the model directory does not")
        check(settings("--first", "3", "--count", "7", "--prompts", "p.jsonl", "--ref", "r.pt", "--label", "AR",
                       "--pair", "o.jsonl:AR", "--transformers", "4", "--threads", "2") == s0,
              "--first, --count, --prompts, --ref, --label, --pair, --transformers and --threads do not")

        done, f = runner.open_results(out, s0)
        check(done == {}, "a new file holds no questions")
        for q in (0, 1):
            f.write(json.dumps({"q": q, "len": 10, "need": 3}) + "\n")
        f.close()
        done, f = runner.open_results(out, s0)
        f.close()
        check(sorted(done) == [0, 1], "the same settings go on after the questions the file holds")
        why = refused(out, settings("--tree", "pool:31:7:1"))
        check(why is not None and "drafter.tree" in why and "drafter.width" not in why,
              f"a DYN run's file is refused to an ST run: {why}")
        why = refused(out, settings("--dtype", "bfloat16", "--canvas", files["canvas2.pt"]))
        check(why is not None and "canvas" in why and "dtype" in why, f"other files and dtype are refused: {why}")
        bare = put("bare.jsonl", b'{"q": 0, "len": 10, "need": 3}\n')
        why = refused(bare, s0)
        check(why is not None and "no settings line" in why, f"rows without their settings are refused: {why}")
        with open(out, encoding="utf-8") as f:
            check(len(f.readlines()) == 3, "nothing was added to the file it refused")

        # A run stopped while writing a row leaves it cut off: refused to a reader, dropped by a run that goes on (its
        # question runs again).  A byte order mark (PowerShell's utf8) is read past; a line in the middle that is not
        # JSON is refused.
        row = lambda q: json.dumps({"q": q, "len": 10, "need": 3, "threads": 20}).encode()  # noqa: E731
        head = json.dumps({"settings": s0}).encode() + b"\n" + row(0) + b"\n"
        cut = put("cut.jsonl", head + row(1)[:12])
        try:
            runner.read_results(cut)
            check(False, "a cut-off last line: refused to a reader")
        except ValueError as e:
            check("cut off" in str(e), f"a cut-off last line: refused to a reader: {e}")
        done, f = runner.open_results(cut, s0)
        f.write(row(1).decode() + "\n")
        f.close()
        check(sorted(done) == [0] and sorted(runner.read_results(cut)[1]) == [0, 1],
              "a cut-off last line: dropped by a run that goes on, its question written again")
        bom = put("bom.jsonl", b"\xef\xbb\xbf" + head + row(1))  # the last row whole, without its newline
        done, f = runner.open_results(bom, s0)
        f.write(row(2).decode() + "\n")
        f.close()
        check(sorted(done) == [0, 1] and sorted(runner.read_results(bom)[1]) == [0, 1, 2],
              "a byte order mark; a whole last row without its newline gets one before the next")
        junk = put("junk.jsonl", head + b"{junk\n" + row(1) + b"\n")
        why = refused(junk, s0)
        check(why is not None and "line 3 is not JSON" in why, f"a line in the middle that is not JSON: {why}")

        # One file's rows come from one thread count.
        held = {0: {"q": 0, "threads": 20}, 1: {"q": 1, "threads": 20}}
        check(runner.threads_refused(out, held, 20) is None and runner.other_threads(held, 8) == [20]
              and "other thread counts (20)" in runner.threads_refused(out, held, 8)
              and "unrecorded" in runner.threads_refused(out, {0: {"q": 0}}, 20),
              "rows made at another thread count, or an unrecorded one, refuse a run at this one")
    print("dv_gsm8k results: " + ("ok" if not failures else f"{failures} failures"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
