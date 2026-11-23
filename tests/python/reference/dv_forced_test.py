# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""python/reference/dv_forced.py: on a synthetic flyx bench record with a stand-in for the model, then whole runs of
it on the tiny Llama's DV (make_tiny.py's assets), in float64.

    python tests/python/reference/dv_forced_test.py ASSETS

* A question is forced to its answer's first L tokens (up to the first end token, at most the record's cap) after
  the record's prompt; its line keeps both, the forwards' a_k, need and flyx's count; a run that emits other tokens
  than it was forced to is refused.  --replay also commits the record's own rounds (its trace's a_k, the last one
  clipped to the answer); a record without a drafter has none to replay.
* Every option that changes the rounds changes the settings the file keeps (the record, the tree, slots, fill, dtype,
  the weights given either way, --replay); --first, --count, --free and --threads do not.  A file goes on only with
  its own settings (another record's, another tree's, a free run's file is refused) and at the thread count its rows
  were made at; a last line cut off by a stopped run is dropped and its question runs again.
* --free: on the questions whose free answer is the record's, the forced and free rounds must agree a_k for a_k, the
  free run's last forward clipped to the answer (it may accept drafts past the end token or the cap); other
  answers are not compared; a free run of another drafter, dtype or thread count is refused, as is a replayed run.
* Whole runs (main): --first and --count choose the record's questions; --threads is the rows' count; the DV of a
  .flyx file and its checkpoints beside the Hugging Face directory (--mask-token the tiny tokenizer's) run the same
  rounds; a V-LoRA's adapters reach the target; --flyx refuses checkpoints beside it; --replay commits the record's
  rounds.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "python", "reference"))
import dv_forced as forced  # noqa: E402
import dv_gsm8k as runner  # noqa: E402
from flyx_ref.dv import DvConfig, flyx_rounds, hf_llama, rounds_needed  # noqa: E402

failures = 0
EOS = (2, 7)


def check(ok, what):
    global failures
    print(("ok " if ok else "FAIL ") + what, flush=True)
    failures += not ok


class Stand:
    """Dv.generate's results for a forced run: the forced tokens, `accepted` drafts a round (as many rounds as the
    answer needs; a round never accepts past the answer's end), the counts given to commit while they last."""

    def __init__(self, accepted, emit=None):
        self.accepted, self.emit, self.calls = accepted, emit, []

    def generate(self, prompt, cap, eos, forced=None, commits=None):
        self.calls.append((list(prompt), cap, tuple(eos), list(forced), commits))
        acc, done, c = [], [], 0
        for k, a in enumerate(self.accepted):
            if c >= len(forced):
                break
            acc.append(min(a, len(forced) - c - 1))
            done.append(commits[k] if commits is not None and k < len(commits) else acc[-1])
            c += 1 + done[-1]
        ids = self.emit if self.emit is not None else list(forced)
        return {"ids": ids, "accepted": acc, "committed": done, "need": rounds_needed(done, len(ids)),
                "flyx_rounds": flyx_rounds(done, len(ids))}


def write(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    return path


def quietly(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args)


def exits(argv):
    """dv_forced.main's exit: its status, or the message it stopped with."""
    try:
        return quietly(forced.main, argv)
    except SystemExit as e:
        return e.code


def rows_of(path):
    return runner.read_results(path)[1]


def synthetic(d):
    def put(name, data):
        path = os.path.join(d, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return path

    # ---- the record ---------------------------------------------------------------------------------------------
    prompts = [{"i": 0, "prompt_ids": [1, 5, 6], "ids": [10, 11, 12, 13, 2], "trace": [[1, 0, 0], [3, 2, 0]]},
               {"i": 1, "prompt_ids": [1, 8], "ids": [20, 21, 22, 23, 24, 25],                 # cut at the cap
                "trace": [[1, 0, 0], [5, 4, 0]]},
               {"i": 3, "prompt_ids": [1, 9], "ids": [30, 7], "trace": [[1, 0, 1]]}]       # the other end token
    record = write(os.path.join(d, "plain.json"), {"eos_ids": list(EOS), "max_new_tokens": 5, "prompts": prompts,
                                                   "drafter": "none"})
    eos, cap, answers = forced.read_record(record)
    check(eos == EOS and cap == 5, "record: its end tokens and cap")
    check(answers[0]["ids"] == [10, 11, 12, 13, 2] and answers[1]["ids"] == [20, 21, 22, 23, 24]
          and answers[3]["ids"] == [30, 7] and answers[1]["prompt_ids"] == [1, 8] and "commits" not in answers[0],
          "record: each answer up to its first end token, at most the cap; the prompt as the record holds it")
    check(forced.answer_length([4, 2, 4, 7], EOS, 9) == 2 and forced.answer_length([4] * 9, EOS, 6) == 6,
          "length: to the first end token, included; at most the cap")
    drafted = write(os.path.join(d, "drafted.json"), {"eos_ids": list(EOS), "max_new_tokens": 5, "prompts": prompts,
                                                      "drafter": "dv"})
    _, _, replays = forced.read_record(drafted, replay=True)
    check(replays[0]["commits"] == [0, 2] and replays[1]["commits"] == [0, 3] and replays[3]["commits"] == [1],
          "record, --replay: each round's a_k (an end token drafted among them), the last clipped to the answer")
    try:
        forced.read_record(record, replay=True)
        check(False, "--replay: a record without a drafter is refused")
    except ValueError as e:
        check("no rounds to replay" in str(e), f"--replay: a record without a drafter is refused: {e}")

    # ---- the settings -------------------------------------------------------------------------------------------
    model = os.path.dirname(put("model/config.json", b"{}"))
    put("model/model.safetensors", b"weights")
    files = {n: put(n, n.encode()) for n in ("canvas.pt", "canvas2.pt", "sel.pt", "sel2.pt", "vlora.pt",
                                             "dv.flyx", "dv2.flyx")}
    record2 = write(os.path.join(d, "plain2.json"), {"eos_ids": list(EOS), "max_new_tokens": 5,
                                                     "prompts": prompts[:2]})
    out = os.path.join(d, "forced.jsonl")
    header = lambda: DvConfig(slots=32, mask_token=99, width=31, tree="pool", pool_r=7, fill=0.9)  # noqa: E731

    def settings(*options, rec=record, flyx=False):
        base = [rec, "--out", out] + (["--flyx", files["dv.flyx"]] if flyx else
                                      ["--model", model, "--canvas", files["canvas.pt"], "--selector",
                                       files["sel.pt"]])
        a = forced.arguments().parse_args(base + list(options))
        dv = forced.drafter_config(a, header() if flyx else None, {"n": 32, "res_k": 16}, 99)
        e, c, _ = forced.read_record(rec)
        return forced.run_settings(a, dv, e, c)
    s0 = settings()
    check(s0["drafter"]["tree"] == "beam" and s0["drafter"]["width"] == 31 and s0["drafter"]["cov"]
          and s0["drafter"]["fill"] == 0.9 and s0["drafter"]["slots"] == 32 and "replay" not in s0,
          "checkpoints: dv_gsm8k.py's defaults (beam:31:8:cov, the canvas's slots, fill 0.9)")
    for option, value in (("--tree", "pool:31:7:1"), ("--tree", "beam:15:8:cov"), ("--slots", "64"),
                          ("--fill", "0.5"), ("--dtype", "float64"), ("--canvas", files["canvas2.pt"]),
                          ("--selector", files["sel2.pt"]), ("--vlora", files["vlora.pt"])):
        check(settings(option, value) != s0, f"{option} {os.path.basename(value)} changes the settings")
    check(settings(rec=record2) != s0, "another record changes the settings")
    check(settings("--replay")["replay"] is True, "--replay changes the settings")
    check(settings("--first", "3", "--count", "7", "--free", "f.jsonl", "--threads", "2") == s0,
          "--first, --count, --free and --threads do not")
    f0 = settings(flyx=True)
    check(f0["drafter"]["tree"] == "pool" and f0["drafter"]["width"] == 31 and f0["canvas"] is None
          and f0["model"] == runner.file_digest(files["dv.flyx"]), "a .flyx file: its header's tree, the file "
          "by its sha256")
    f15 = settings("--tree", "pool:15:7:1", flyx=True)
    check(f15["drafter"]["width"] == 15 and f15["drafter"]["pool_r"] == 7 and f15 != f0,
          "a .flyx file: --tree over its header's (pool 15/7/1, as flyx's 15-draft rounds draft)")
    check(settings("--fill", "0.5", flyx=True)["drafter"]["fill"] == 0.5
          and settings("--slots", "64", flyx=True)["drafter"]["slots"] == 64, "a .flyx file: --fill, --slots")
    a = forced.arguments().parse_args([record, "--out", out, "--flyx", files["dv2.flyx"]])
    check(forced.run_settings(a, header(), eos, cap) != f0, "another .flyx file changes the settings")

    # ---- the run ------------------------------------------------------------------------------------------------
    done, f = runner.open_results(out, s0)
    check(done == {}, "a new file holds no questions")
    stand = Stand([0, 2, 1, 3, 3])
    rows = quietly(forced.force, stand, answers, [0, 1], eos, cap, done, f, 20)
    f.close()
    r0 = rows[0]
    check(stand.calls[0] == ([1, 5, 6], 5, EOS, [10, 11, 12, 13, 2], None),
          "forced: the record's prompt, its cap and end tokens, the answer's first L tokens")
    check(r0["len"] == 5 and r0["accepted"] == [0, 2, 0] and r0["need"] == 3 and r0["flyx_rounds"] == 2
          and r0["new"] == 4 and r0["ended"] and r0["ids"] == [10, 11, 12, 13, 2]
          and r0["prompt_ids"] == [1, 5, 6] and r0["threads"] == 20 and "committed" not in r0,
          f"a row: L, a_k, need, flyx's rounds, new, ended, the answer and prompt, threads: {r0}")
    check(rows[1]["len"] == 5 and not rows[1]["ended"] and rows[1]["new"] == 5, "a row: an answer cut at the cap")
    done, f = runner.open_results(out, s0)
    stand2 = Stand([0, 1, 1, 1, 1, 1])
    rows2 = quietly(forced.force, stand2, answers, [0, 1, 3], eos, cap, done, f, 4)
    f.close()
    check(len(stand2.calls) == 1 and rows2[:2] == rows and rows2[2]["q"] == 3,
          "the same settings go on after the questions the file holds")
    for why_s, what in ((settings("--tree", "pool:31:7:1"), "another tree's"),
                        (settings(rec=record2), "another record's")):
        try:
            runner.open_results(out, why_s)[1].close()
            check(False, f"{what} file is refused")
        except ValueError as e:
            check("give another --out" in str(e), f"{what} file is refused: {e}")
    free_settings = {k: v for k, v in s0.items() if k not in ("forced", "eos")} | {"questions": "q"}
    free_like = put("free.jsonl", json.dumps({"settings": free_settings}).encode() + b"\n")
    try:
        runner.open_results(free_like, s0)[1].close()
        check(False, "a free run's file is refused")
    except ValueError as e:
        check("forced" in str(e) and "questions" in str(e), f"a free run's file is refused: {e}")
    try:
        quietly(forced.force, Stand([0, 1], emit=[10, 11]), answers, [0], eos, cap, {}, io.StringIO(), 1)
        check(False, "a run that emits other tokens than the record's is refused")
    except RuntimeError as e:
        check("not emit the record's answer" in str(e),
              f"a run that emits other tokens than the record's is refused: {e}")
    held = runner.read_results(out)[1]
    check(runner.threads_refused(out, held, 20) is not None and runner.threads_refused(out, {0: held[0]}, 20) is None
          and "unrecorded" in runner.threads_refused(out, {0: {"q": 0}}, 20),
          "rows made at another thread count, or at an unrecorded one, refuse a run at 20")

    # ---- replayed -----------------------------------------------------------------------------------------------
    replayer = Stand([0, 0, 0, 0, 0])
    rrows = quietly(forced.force, replayer, replays, [0, 3], eos, cap, {}, io.StringIO(), 2)
    check(replayer.calls[0][4] == [0, 2] and rrows[0]["committed"] == [0, 2, 0] and rrows[0]["replayed"] == 2
          and rrows[0]["accepted"] == [0, 0, 0] and rrows[0]["need"] == 3,
          f"--replay: the record's counts committed, the forward's own a_k kept; need of what it committed: {rrows[0]}")
    line = forced.against_record(rrows)
    check("own a_k the record's in 1 of 3 rounds" in line and "drafts accepted 0 here, 3 in the record" in line
          and "every round the same in 0 of 2" in line,
          f"--replay: its a_k against the record's rounds (its last own forward not counted): {line}")
    class Unreplayed(Stand):
        def generate(self, prompt, cap, eos, forced=None, commits=None):
            return Stand.generate(self, prompt, cap, eos, forced)
    try:
        quietly(forced.force, Unreplayed([0, 0, 0]), replays, [0], eos, cap, {}, io.StringIO(), 1)
        check(False, "--replay: a run that commits other counts than the record's is refused")
    except RuntimeError as e:
        check("did not commit the record's rounds" in str(e),
              f"--replay: a run that commits other counts than the record's is refused: {e}")

    # ---- against a free run -------------------------------------------------------------------------------------
    check(forced.answer_rounds([0, 5], 4) == [0, 2] and forced.answer_rounds([0, 2, 1, 4], 5) == [0, 2, 0]
          and forced.answer_rounds([0, 1], 9) == [0, 1] and forced.answer_rounds([3], 1) == [0],
          "answer_rounds: the last forward clipped to the answer; forwards after it dropped; short runs kept")
    rows = rows2

    def free_file(name, free_rows, held=free_settings):
        lines = [json.dumps({"settings": held})] + [json.dumps(r) for r in free_rows]
        return put(name, ("\n".join(lines) + "\n").encode())
    # question 0: the same answer, the free run's last forward accepted drafts past the end token; question 1:
    # another answer (not compared); question 3: the same answer and rounds
    agree_rows = [{"q": 0, "len": 5, "ids": [10, 11, 12, 13, 2], "accepted": [0, 2, 4], "need": 3},
                  {"q": 1, "len": 5, "ids": [20, 21, 99, 23, 24], "accepted": [0, 0, 0, 0, 0], "need": 5},
                  {"q": 3, "len": 2, "ids": [30, 7], "accepted": rows[2]["accepted"], "need": rows[2]["need"]}]
    lines, ok = forced.against_free(rows, s0, free_file("agree.jsonl", agree_rows))
    check(ok and "3 questions in both, 2 with the same answer" in lines[0] and "2/2 questions" in lines[0]
          and "thread count unrecorded" in lines[0],
          f"--free: the same rounds where the answers agree, the free last forward clipped: {lines}")
    lines, ok = forced.against_free(rows, s0, free_file("agree_t.jsonl", [dict(r, threads=t) for r, t in
                                                                          zip(agree_rows, (20, 20, 4))]))
    check(ok and "threads 20, 4" in lines[0], f"--free: made at the same thread counts: {lines}")
    lines, ok = forced.against_free(rows, s0, free_file("other_t.jsonl", [dict(r, threads=t) for r, t in
                                                                          zip(agree_rows, (20, 20, 20))]))
    check(not ok and "1 questions made at another thread count than here (first: question 3, 20 threads against 4)"
          in lines[0], f"--free: a question made at another thread count: refused: {lines}")
    part = free_file("part.jsonl", [{"q": 0, "len": 5, "ids": [10, 11, 12, 13, 2], "accepted": [0, 1, 1, 0],
                                     "need": 4}])
    lines, ok = forced.against_free(rows, s0, part)
    check(not ok and any("question 0: the rounds part at forward 2" in s for s in lines),
          f"--free: a round with another a_k fails, where they part said: {lines}")
    none = free_file("none.jsonl", [{"q": 1, "len": 5, "ids": [20, 21, 99, 23, 24], "accepted": [0] * 5,
                                     "need": 5}])
    check(not forced.against_free(rows, s0, none)[1], "--free: no answer in common fails")
    for key, value in (("drafter", dict(s0["drafter"], width=15)), ("dtype", "bfloat16")):
        lines, ok = forced.against_free(rows, s0, free_file(f"other{len(key)}.jsonl", agree_rows,
                                                            dict(free_settings, **{key: value})))
        check(not ok and "made with other settings" in lines[0] and key in lines[0],
              f"--free: another {key}'s run is refused: {lines}")
    lines, ok = forced.against_free(rows, dict(s0, replay=True), free_file("agree2.jsonl", agree_rows))
    check(not ok and "--replay" in lines[0], f"--free: a replayed run is refused: {lines}")


def whole_runs(d, assets):
    """dv_forced.main on the tiny Llama's DV, float64."""
    golden = json.load(open(os.path.join(assets, "golden_llama.json"), encoding="utf-8"))
    cases = golden["cases"][:4]
    n = len(cases[0]["tokens"])
    prompts = [{"i": i, "prompt_ids": c["prompt"], "ids": c["tokens"], "trace": [[1, 0, 0]] * (n - 1)}
               for i, c in enumerate(cases)]
    plain = write(os.path.join(d, "tiny_plain.json"), {"eos_ids": [0, 2], "max_new_tokens": n, "prompts": prompts,
                                                       "drafter": "none"})
    # a drafted run's rounds: 0, then 2, 0, 3, 1, ... (what the tiny DV's random canvas would never accept itself)
    trace = []
    while sum(t[0] for t in trace) < n - 1:
        a = (0, 2, 0, 3, 1)[len(trace) % 5]
        trace.append([min(a + 1, n - 1 - sum(t[0] for t in trace)), a, 0])
    drafted = write(os.path.join(d, "tiny_dv.json"), {"eos_ids": [0, 2], "max_new_tokens": n, "drafter": "dv",
                                                      "prompts": [dict(p, trace=trace) for p in prompts]})
    pool, beam = os.path.join(assets, "tiny_llama_dv_pool.flyx"), os.path.join(assets, "tiny_llama_dv.flyx")
    ck = ["--model", os.path.join(assets, "hf_llama"), "--canvas", os.path.join(assets, "dv_tiny", "canvas.pt"),
          "--selector", os.path.join(assets, "dv_tiny", "selector.pt"), "--mask-token", "<|mask|>"]
    # The pool file's drafter: the causal canvas at fill 0.4 (make_tiny.py's export_pool).
    ck_pool = ck[:2] + ["--canvas", os.path.join(assets, "dv_tiny", "canvas_causal.pt"), "--fill", "0.4"] + ck[4:]
    out = lambda name: os.path.join(d, name)  # noqa: E731

    code = exits([plain, "--flyx", pool, "--dtype", "float64", "--threads", "2", "--first", "1", "--count", "2",
                  "--out", out("flyx.jsonl")])
    by_flyx = rows_of(out("flyx.jsonl"))
    check(code == 0 and sorted(by_flyx) == [1, 2] and all(r["threads"] == 2 for r in by_flyx.values())
          and torch.get_num_threads() == 2 and all(r["ids"] == cases[q]["tokens"] for q, r in by_flyx.items()),
          f"main: --first 1 --count 2 forces questions 1 and 2 (of 0 to 3) at --threads 2 ({code})")
    code = exits([plain] + ck_pool + ["--tree", "pool:15:3:1", "--dtype", "float64", "--threads", "2", "--first", "1",
                                 "--count", "2", "--out", out("ck.jsonl")])
    by_ck = rows_of(out("ck.jsonl"))
    check(code == 0 and {q: r["accepted"] for q, r in by_ck.items()} == {q: r["accepted"] for q, r in by_flyx.items()}
          and runner.read_results(out("ck.jsonl"))[0]["drafter"]
          == runner.read_results(out("flyx.jsonl"))[0]["drafter"],
          f"main: the .flyx file's DV and its checkpoints run the same rounds "
          f"({sum(len(r['accepted']) for r in by_ck.values())} forwards)")
    check(exits([plain] + ck + ["--dtype", "float64", "--count", "1", "--out", out("ck_beam.jsonl")]) == 0
          and exits([plain, "--flyx", beam, "--tree", "beam:31:8:cov", "--dtype", "float64", "--count", "1",
                     "--out", out("flyx_beam.jsonl")]) == 0
          and rows_of(out("ck_beam.jsonl"))[0]["accepted"] == rows_of(out("flyx_beam.jsonl"))[0]["accepted"],
          "main: the same, beam 31:8:cov (the checkpoints' default tree)")
    why = exits([plain, "--flyx", pool, "--canvas", os.path.join(assets, "dv_tiny", "canvas.pt"), "--out",
                 out("x.jsonl")])
    check(isinstance(why, str) and "go with --model" in why, f"main: --flyx with a canvas beside it: refused: {why}")
    why = exits([plain] + ck[:6] + ["--mask-token", "<no such token>", "--out", out("x.jsonl")])
    check(isinstance(why, str) and "not a token" in why, f"main: a MASK token the tokenizer lacks: refused: {why}")

    # a V-LoRA's adapters reach the target's weights
    cfg, _ = hf_llama(os.path.join(assets, "hf_llama"))
    gen = torch.Generator().manual_seed(3)
    lora = [{"A": torch.randn(2, 8, generator=gen), "B": torch.randn(8, 2, generator=gen)}
            for _ in range(7 * cfg.layers)]
    torch.save({"lora": lora}, out("vlora.pt"))
    a = forced.arguments().parse_args([plain] + ck + ["--vlora", out("vlora.pt"), "--out", out("x.jsonl")])
    _, weights = forced.assets(a)
    _, W = weights()
    check(torch.equal(W["target.L0.attn.q.lora_a"], lora[0]["A"].to(torch.bfloat16))
          and torch.equal(W[f"target.L{cfg.layers - 1}.mlp.down.lora_b"], lora[-1]["B"].to(torch.bfloat16)),
          "main: --vlora's adapters are the target's (rounded to bfloat16)")
    # ... and a .flyx file's V-LoRA is in the settings (check_dv.py --oracle ties the oracle's target to the record's)
    with_lora = os.path.join(assets, "tiny_llama_dv_lora.flyx")
    settings = {}
    for f in (pool, with_lora):
        b = forced.arguments().parse_args([plain, "--flyx", f, "--out", out("x.jsonl")])
        settings[f] = forced.run_settings(b, forced.assets(b)[0], (2,), 48)["vlora"]
    sha = json.load(open(os.path.join(assets, "dv_tiny", "export_dv_lora", "vlora.json"), encoding="utf-8"))
    check(settings[pool] is None and settings[with_lora] == sha["source"]["sha256"],
          f"run_settings: a .flyx file's V-LoRA by its source's sha256 ({settings})")

    # a run goes on at its rows' thread count; a cut-off last line runs again
    why = exits([plain, "--flyx", pool, "--dtype", "float64", "--threads", "3", "--first", "0", "--count", "2",
                 "--out", out("flyx.jsonl")])
    check(isinstance(why, str) and "other thread counts (2)" in why, f"main: a run at another thread count: {why}")
    with open(out("flyx.jsonl"), "rb") as f:
        whole = f.read()
    with open(out("flyx.jsonl"), "wb") as f:
        f.write(whole[:len(whole) - 30])
    code = exits([plain, "--flyx", pool, "--dtype", "float64", "--threads", "2", "--first", "1", "--count", "2",
                  "--out", out("flyx.jsonl")])
    check(code == 0 and rows_of(out("flyx.jsonl")).keys() == by_flyx.keys()
          and {q: r["accepted"] for q, r in rows_of(out("flyx.jsonl")).items()}
          == {q: r["accepted"] for q, r in by_flyx.items()},
          f"main: a last line cut off is dropped and its question runs again ({code})")

    # --replay: every forward commits the drafted record's counts
    code = exits([drafted, "--flyx", pool, "--replay", "--dtype", "float64", "--threads", "2", "--count", "2",
                  "--out", out("replay.jsonl")])
    replayed = rows_of(out("replay.jsonl"))
    want = forced.answer_rounds(forced.record_rounds(trace), n)
    check(code == 0 and all(r["committed"][:r["replayed"]] == want and r["replayed"] == len(want)
                            and r["ids"] == cases[q]["tokens"] for q, r in replayed.items())
          and runner.read_results(out("replay.jsonl"))[0]["replay"] is True,
          f"main: --replay commits the record's rounds ({len(want)} a question) ({code})")
    why = exits([plain, "--flyx", pool, "--replay", "--out", out("x.jsonl")])
    check(isinstance(why, str) and "no rounds to replay" in why, f"main: --replay on plain decoding's record: {why}")
    why = exits([drafted, "--flyx", pool, "--replay", "--free", out("flyx.jsonl"), "--out", out("x.jsonl")])
    check(isinstance(why, str) and "--free" in why, f"main: --replay with --free: refused: {why}")


def main(assets):
    with tempfile.TemporaryDirectory() as d:
        synthetic(d)
        whole_runs(d, assets)
    print("dv_forced: " + ("ok" if not failures else f"{failures} failures"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
