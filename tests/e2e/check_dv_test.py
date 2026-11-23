# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""check_dv.py and dv_ref_json.py: an answer's length, the forwards it needs and tokens per forward as bifill-dv
counts them (an end token accepted as a draft among them), GSM8K answers scored as bifill-dv scores them, a reference
made from the three layouts of bifill-dv's result files and its stored verdicts re-scored, two bench records held to
each other (the plain one plain decoding), speed and the reference by prompt range, a drafted record's rounds against
the teacher-forced reference's (python/reference/dv_forced.py: refused unless forced to the record's answers with the
DV its model file holds; round for round only when it replayed the record's rounds), and the files a run is checked
on refused when they cannot be read.  No model, no torch (a .flyx file of a header alone stands for the record's
model file).

    python tests/e2e/check_dv_test.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import check_dv as cd  # noqa: E402
import dv_ref_json  # noqa: E402

failures = []
BS = chr(92)  # a backslash


def main_quietly(argv):
    """check_dv.py's exit status for these arguments, its report left out."""
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            return cd.main(argv)
        except SystemExit as ex:
            return ex.code


def check(ok, what, got=None):
    print(f"  {'ok' if ok else 'FAIL':4s} {what}" + ("" if ok or got is None else f": got {got!r}"))
    if not ok:
        failures.append(what)


# ---- an answer's length --------------------------------------------------------------------------------------------
E = 130073
check(cd.answer_length([5, 6, E, 7]) == 3, "length: up to the first end token, included")
check(cd.answer_length([5, 1, 6]) == 2, "length: either end token (</s> 1, <|im_end|> 130073)")
check(cd.answer_length([5] * 600) == 512, "length: at most 512")
check(cd.answer_length([5] * 511 + [E]) == 512, "length: an end token as token 512 counts")
check(cd.answer_length([5] * 512 + [E]) == 512, "length: one past 512 does not")
check(cd.answer_length([5] * 40) == 40, "length: an answer cut by its limit, all of it")
check(cd.answer_length([7, 9, 2], eos=(2,), max_new=8) == 3, "length: the record's end tokens and limit")
check(cd.ended([5, E]) and not cd.ended([5, 6]), "ended: an end token in it")

# ---- forwards needed -----------------------------------------------------------------------------------------------
check(cd.forwards_needed(180, [0] * 179) == 180, "forwards: plain decoding, the end token's round the last: L")
check(cd.forwards_needed(180, [0] * 181) == 180, "forwards: rounds past the end change nothing")
check(cd.forwards_needed(1, []) == 1, "forwards: the first token the end: 1 (it is committed by forward 1)")
check(cd.forwards_needed(10, [0, 3, 4, 2]) == 3, "forwards: DV, the first round all MASK (commits 1, 5, 10)")
check(cd.forwards_needed(11, [0, 3, 4, 2]) == 4, "forwards: one token more, one forward more")
check(cd.forwards_needed(20, [0, 1]) == 3, "forwards: never there in R rounds: R + 1")
# bifill-dv's own formula (its mega engine, bifill/mega/engine.py): int((cumsum(1 + a) < len).sum()) + 1, on random
# answers
rng = random.Random(7)
same = True
for _ in range(2000):
    acc = [0] + [rng.choice((0, 0, 1, 2, 3, 5, 8, 15, 31)) for _ in range(rng.randrange(0, 60))]
    L = rng.randrange(1, 300)
    total, below = 0, 0
    for a in acc:
        total += 1 + a
        below += total < L
    same &= cd.forwards_needed(L, acc) == below + 1
check(same, "forwards: bifill-dv's cumsum formula on 2000 random answers")

rows = [{"len": 100, "need": 20, "new": 99, "rounds": 21}, {"len": 50, "need": 10, "new": 50, "rounds": 10}]
check(abs(cd.tpf_bifill(rows) - 150 / 30) < 1e-12, "TPF, bifill-dv's: sum L / sum forwards")
check(abs(cd.tpf_flyx(rows) - (98 + 49) / 31) < 1e-12, "TPF, flyx bench's: sum (new - 1) / sum rounds")


def prompt(i, ids, trace, decode_ms=1.0, prefill_ms=1.0, **kw):
    """A prompt of a bench record: its answer's tokens and trace ([tokens, accepted drafts, last token a draft])."""
    stopped = ids[-1] == 2
    return dict({"i": i, "ids": ids, "trace": trace, "new_tokens": len(ids) - stopped, "rounds": len(trace),
                 "decode_ms": decode_ms, "prefill_ms": prefill_ms, "drafted": 0, "prompt_ids": [1, 2]}, **kw)


record = {"eos_ids": [2], "max_new_tokens": 16, "prompts": [
    prompt(0, [5, 6, 7, 8, 2], [[2, 1, 0], [2, 1, 0]]),
    prompt(1, [5] * 16, [[1, 0, 0]] * 15),
    # the prefill gives 5; one round accepts the drafts 6 and 7, then the draft 2, an end token: flyx's ring counts it
    # among the round's tokens, not its accepted drafts; bifill-dv's forward 1 commits 5 and three drafts, all of L
    prompt(2, [5, 6, 7, 2], [[3, 2, 1]]),
    # the same tokens, the 2 the target's own after two drafts: forward 2 commits it (bifill-dv)
    prompt(3, [5, 6, 7, 2], [[3, 2, 0]]),
    # a record from before the third value: the end token's round has other tokens, so it cannot tell
    prompt(4, [5, 6, 7, 2], [[3, 2]]),
    prompt(5, [5, 6, 2], [[1, 0], [1, 0]])]}
rows = cd.per_prompt(record)
check(rows[0]["len"] == 5 and rows[0]["need"] == 3 and rows[0]["ended"], "per prompt: the record's end tokens",
      rows[0])
check(rows[1]["len"] == 16 and rows[1]["need"] == 16 and not rows[1]["ended"], "per prompt: cut by the limit",
      rows[1])
check(rows[2]["need"] == 1, "per prompt: an end token accepted as a draft is one of the round's a_k (forward 1, "
      "the prefill not a forward)", rows[2])
check(rows[3]["need"] == 2, "per prompt: the target's own end token after the drafts: the next forward", rows[3])
check(cd.accepted_by_round([[3, 2, 1], [1, 0, 0], [4, 3]]) == [3, 0, 3], "a_k: accepted drafts, the end token "
      "drafted one more; two values read as before")
check(rows[4]["need"] == 2 and rows[4]["unknown"] and not rows[3]["unknown"] and not rows[5]["unknown"],
      "per prompt: an old record's end token in a round with other tokens: counted the round's own, flagged", rows[4])
check(rows[2]["ids"] == [5, 6, 7, 2] and cd.per_prompt({"eos_ids": [2], "max_new_tokens": 3, "prompts": [
      prompt(0, [5, 6, 7, 8], [[1, 0, 0]] * 3)]})[0]["ids"] == [5, 6, 7], "per prompt: the answer up to L")
check(cd.default_tolerance(1319) == 1.5 and cd.default_tolerance(500) == 2 and cd.default_tolerance(100) == 3,
      "tolerance by questions: 1.5% from 1000, 2% from 500, else 3%")

# ---- speed by prompt range -----------------------------------------------------------------------------------------
speed = {i: {"new": 101, "decode_ms": ms} for i, ms in enumerate((400.0, 500.0, 1000.0))}  # 250, 200, 100 tok/s
check(cd.parse_ranges("0:2,1:3") == [(0, 2), (1, 3)], "ranges: A:B, comma-separated")
lines = cd.speed_ranges(speed, [(0, 2), (0, 3)])
check(lines == ["prompts 0..1 (2): 222.2 tok/s (median prompt 225.0, slowest 200.0)",
                "prompts 0..2 (3): 157.9 tok/s (median prompt 200.0, slowest 100.0)"],
      "ranges: tokens over decode time, the median prompt's rate and the slowest", lines)
check(cd.median([3.0, 1.0, 2.0]) == 2.0 and cd.median([4.0, 1.0]) == 2.5, "median")

# ---- GSM8K as bifill-dv scores it ----------------------------------------------------------------------------------
check(cd.gsm_gold("She makes 9 * 2 = $<<9*2=18>>18.\n#### 18") == "18", "gold: after ####")
check(cd.gsm_gold("#### 1,234 ") == "1234", "gold: thousands' commas dropped")
check(cd.gsm_pred("So she makes 18 dollars.") == "18", "pred: the last number")
check(cd.gsm_pred(BS + "boxed{18} for 2 days") == "2",
      "pred: bifill-dv's pattern never finds LaTeX's \\boxed{}: the text's last number",
      cd.gsm_pred(BS + "boxed{18} for 2 days"))
check(cd.gsm_pred("it is oxed{7} and 9") == "7", "pred: ... and finds 'oxed{' after a non-word character")
check(cd.gsm_pred("<think>12 apples</think>\n\nthe answer is 5") == "5", "pred: only the text after </think>")
check(cd.gsm_pred("<think>12</think> none") is None, "pred: no number after </think>: none")
check(cd.gsm_pred("total $1,000.00") == "1000", "pred: commas and a trailing .00 dropped")
check(cd.gsm_pred("3.0 and 2.50") == "2.50", "pred: only .00 and .0 are dropped")
check(cd.gsm_pred("x = 3.0") == "3" and cd.gsm_pred("lost -4") == "-4", "pred: .0 dropped; a sign kept")
check(cd.gsm_pred("no digits") is None, "pred: no number: none")
check(cd.gsm_correct([5, E], "it is 18", "18"), "correct: ended and the number right")
check(not cd.gsm_correct([5, 6], "it is 18", "18"), "correct: an answer that did not end never counts")

# ---- the reference -------------------------------------------------------------------------------------------------
mega = {"outs": {"AR": [[5, 6, E], [5] * 520], "lossless@0.9": [[5, 6, 7, E], [5] * 512]},
        "perq": {"AR": {"need": [3, 512], "ok": [1.0, 0.0]}, "lossless@0.9": {"need": [2, 100], "ok": [1.0, 0.0]}},
        "cfg": {"N": 2}, "runs": [{"name": "mega sel lossless@0.9", "tpf": (4 + 512) / 102, "ok": 1}]}
ref = dv_ref_json.reference(mega, "lossless@0.9", "mega.pt")
check(ref["questions"]["0"] == {"len": 4, "need": 2, "correct": True}, "ref: bifill-dv's mega engine's layout",
      ref["questions"]["0"])
check(ref["questions"]["1"]["len"] == 512 and ref["aggregate"]["need"] == 102 and ref["aggregate"]["correct"] == 1,
      "ref: lengths, forwards and correct answers summed", ref["aggregate"])
check(ref["reported"] is not None and ref["reported"]["name"] == "mega sel lossless@0.9",
      "ref: the file's own summary found by its numbers")
ar = dv_ref_json.reference(mega, "AR", "mega.pt")
check(ar["questions"]["1"]["len"] == 512 and ar["reported"] is None, "ref: an AR answer of 520 tokens counts 512")
fused = {"outs": {"DVDT@0.9": [[5, E]]}, "per_q": {"DVDT@0.9": {"need": [1], "correct": [False], "dis": [0]}},
         "env": {"VLORA": None}, "runs": [{"label": "DVDT@0.9", "tpf": 2.0}]}
ref2 = dv_ref_json.reference(fused, "DVDT@0.9", "dyn.pt")
check(ref2["questions"]["0"] == {"len": 2, "need": 1, "correct": False} and ref2["reported"]["tpf"] == 2.0
      and ref2["settings"] == {"VLORA": None}, "ref: dv_fused's layout", ref2)

check("ids" not in ref["questions"]["0"], "ref: no answers' tokens unless asked")
with_ids = dv_ref_json.reference(mega, "AR", "mega.pt", ids=True)
check(with_ids["questions"]["0"]["ids"] == [5, 6, E] and len(with_ids["questions"]["1"]["ids"]) == 512,
      "ref --ids: each answer's first L tokens")
rows_ar = [{"i": 1, "resp": [5] * 520, "correct": False, "response_text": "x", "pred": None, "gold": "4"},
           {"i": 0, "resp": [5, 6, E], "correct": True, "response_text": "it is 18", "pred": "18", "gold": "18"}]
gen = dv_ref_json.gen_ar(rows_ar)
ref_gen = dv_ref_json.reference(gen, "AR", "ar_test.jsonl")
check(ref_gen["questions"]["0"] == {"len": 3, "need": 3, "correct": True} and ref_gen["questions"]["1"]["need"] == 512
      and ref_gen["aggregate"]["tpf"] == 1.0, "ref: gen_ar's rows by 'i', plain decoding: forwards = L",
      ref_gen["questions"]["0"])


def fake_decode(ids):
    return {(5, 6, E): "it is 18"}.get(tuple(ids), "x")


lines, differ = dv_ref_json.check_scores(gen, ["AR"], ["18", "4"], fake_decode, "ar_test.jsonl")
check(differ == 0 and lines[0] == "ar_test.jsonl [AR]: correct 1 here, 1 stored, 0 differ"
      and "text differs 0, pred differs 0, gold differs 0" in lines[1], "re-scoring: the verdicts as stored", lines)
gen["rows"][1]["pred"] = "17"
gen["per_q"]["AR"]["correct"][1] = True
lines, differ = dv_ref_json.check_scores(gen, ["AR"], ["18", "4"], fake_decode, "ar_test.jsonl")
check(differ == 2 and "1 differ" in lines[0] and "pred differs 1" in lines[1], "re-scoring: a verdict and a number "
      "that differ are counted", lines)
lines, differ = dv_ref_json.check_scores(mega, ["AR", "lossless@0.9"], ["18", "4"], fake_decode, "mega.pt")
check(differ == 1 and lines[1] == "mega.pt [lossless@0.9]: correct 0 here, 1 stored, 1 differ",
      "re-scoring: the mega layout's ok", lines)

same, n, prefix = cd.identical_answers({0: [1, 2, 3], 1: [1, 2], 2: [9]}, {0: [1, 2, 3], 1: [1, 3], 3: [9]})
check((same, n, prefix) == (1, 2, 2.0), "identical answers: over the questions both have; the common prefix",
      (same, n, prefix))

ours = {0: {"len": 4, "need": 2, "correct": True}, 1: {"len": 512, "need": 101, "correct": False}}
lines, ok = cd.compare_ref(ours, ref)
check(ok and any("-0.97%" in s for s in lines), "against a reference: 1% off, tolerance 3%", lines)
lines, ok = cd.compare_ref(ours, ref, tol=0.5)
check(not ok, "against a reference: outside the tolerance fails", lines)
lines, ok = cd.compare_ref({0: {"len": 3, "need": 1, "correct": True}}, ar)
check(ok and any("accuracy only" in s for s in lines), "against plain decoding: accuracy only", lines)
lines, ok = cd.compare_ref({7: {"len": 3, "need": 1}}, ref)
check(not ok, "against a reference: no question in common fails")
lines, ok = cd.compare_ref({0: {"len": 3, "need": 3, "ids": [5, 6, E]}, 1: {"len": 512, "need": 512, "ids": [6] * 512}},
                           with_ids)
check(ok and any("identical token for token (up to the end token): 1 of 2" in s for s in lines),
      "against a reference with the answers' tokens: those identical counted", lines)
# the first N questions of a reference of them all: question 0 alone, 2 forwards against the reference's 2 (TPF 2
# both); question 1 alone, 101 against 100 (TPF 5.07 against 5.12)
parts = cd.compare_ref_ranges(ours, ref, [(0, 1), (1, 2), (0, 2), (0, 9)])
check([(r, ok) for r, _, ok in parts] == [((0, 1), True), ((1, 2), True)]
      and any("1 questions in common" in s for s in parts[0][1]) and any("+0.00%" in s for s in parts[0][1])
      and any("-0.99%" in s for s in parts[1][1]),
      "against a reference, by range: each range's questions alone; a range of every question left out", parts)
parts = cd.compare_ref_ranges(ours, ref, [(0, 1), (1, 2)], tol=0.5)
check([ok for _, _, ok in parts] == [True, False], "against a reference, by range: a range outside the tolerance "
      "fails", parts)

# ---- two records ---------------------------------------------------------------------------------------------------
base = {"model": "a/m.flyx", "backend": "cuda", "kv": "16", "max_new_tokens": 512, "ignore_eos": False,
        "suffix": "s", "eos_ids": [2], "sampling": {"temperature": 0}, "backend_options": "", "ctx": 2048,
        "command": ["flyx", "bench", "a/m.flyx", "--prompts", "p", "--kv", "16", "--json", "d.json"],
        "prompts": [prompt(0, [5, 2], [[1, 0, 0]], drafted=4)]}
plain = dict(base, model="m.flyx", command=base["command"][:-2] + ["--no-spec", "--no-lookup", "--json", "p.json"],
             prompts=[prompt(0, [5, 2], [[1, 0, 0]])])
fails, warns = cd.compare_records(base, plain)
check(not fails and not warns, "records: the drafter's options and the model's directory may differ", fails + warns)
fails, _ = cd.compare_records(base, dict(plain, kv="fp8"))
check(fails and "kv" in fails[0], "records: another K/V cache fails", fails)
fails, _ = cd.compare_records(base, dict(plain, prompts=[prompt(0, [5, 2], [[1, 0, 0]], prompt_ids=[1, 3])]))
check(fails and "tokens differ" in fails[0], "records: other prompt tokens fail", fails)
fails, _ = cd.compare_records(base, dict(plain, prompts=[prompt(0, [5, 2], [[1, 0, 0]], drafted=1)]))
check(fails and "not plain decoding" in fails[0], "records: a plain record whose rounds drafted fails", fails)
fails, _ = cd.compare_records(base, dict(plain, model="a/other.flyx"))
check(fails and "models differ" in fails[0], "records: another model file fails", fails)
other = dict(plain, model="a/other.flyx", command=[a if a != "a/m.flyx" else "a/other.flyx" for a in plain["command"]])
fails, warns = cd.compare_records(base, other, same_target=True)
check(not fails and len(warns) == 1 and "models differ" in warns[0],
      "records: another file of the same target (--same-target) is shown", fails + warns)
# The drafted record's drafter drafted: DV's rounds counted; a record naming DV with none (written before the engine
# said a refused drafter is not in use), or saying the backend refused it, fails.
dv_ran = dict(base, drafter="dv", drafter_refused="", prompts=[prompt(0, [5, 2], [[1, 0, 0]], drafter_rounds=[1])])
check(cd.drafter_ran(dv_ran) == ([], []), "drafter: DV's rounds ran")
check(cd.drafter_ran(dict(base, drafter="none", drafter_refused="")) == ([], []), "drafter: none, nothing to run")
fails, _ = cd.drafter_ran(dict(dv_ran, prompts=[prompt(0, [5, 2], [[1, 0, 0]], drafter_rounds=[])]))
check(len(fails) == 1 and "drafted no round" in fails[0], "drafter: DV named, no round of it, fails", fails)
refused = dict(base, drafter="none", drafter_refused="dv 15: lower: not here", lookup=True)
fails, warns = cd.drafter_ran(refused)
check(len(fails) == 1 and "refused" in fails[0] and not warns, "drafter: refused by the backend fails", fails)
fails, warns = cd.drafter_ran(refused, refused_ok=True)
check(not fails and len(warns) == 1 and "dv 15" in warns[0], "drafter: refused, --refused-ok: a warning", warns)
fails, _ = cd.drafter_ran(dict(dv_ran, drafter_refused="dv 31: lower: 32 rows"))
check(len(fails) == 1 and "dv 31" in fails[0], "drafter: one width of it refused fails", fails)
_, warns = cd.compare_records(base, dict(plain, command=plain["command"] + ["--ctx", "4096"]))
check(len(warns) == 1, "records: another option on the command line is shown", warns)
check(cd.options_without_drafter({"command": ["f", "bench", "m", "--json", "--drafter", "dv", "-n", "8"]})
      == ["bench", "m", "-n", "8"], "records: --json without a file")

# ---- the teacher-forced oracle (dv_forced.py) --------------------------------------------------------------------
check(cd.answer_rounds([0, 5], 4) == [0, 2] and cd.answer_rounds([0, 2, 1, 4], 5) == [0, 2, 0]
      and cd.answer_rounds([0, 1], 9) == [0, 1] and cd.answer_rounds([3], 1) == [0],
      "answer rounds: the last forward clipped to the answer; forwards after it dropped; short runs kept")
# Question 0: L 10; flyx's last round ends with the target's own end token, so bifill-dv's last forward (the one that
# commits it as its anchor) is not in the trace; the oracle has it (a_k 0).  Question 1: L 6; the end token accepted
# as a draft; the oracle accepted one draft fewer in round 2 and one more in round 3.  Question 2: not in the oracle.
dv_record = dict(base, drafter="dv", drafter_refused="", widths=[15], max_new_tokens=16, prompts=[
    prompt(0, [10, 11, 12, 13, 14, 15, 16, 17, 18, 2], [[1, 0, 0], [4, 3, 0], [4, 3, 0]], prompt_ids=[1, 5],
           drafter_rounds=[3]),
    prompt(1, [20, 21, 22, 23, 24, 2], [[1, 0, 0], [3, 2, 0], [1, 0, 1]], prompt_ids=[1, 6], drafter_rounds=[3]),
    prompt(2, [30, 2], [[1, 0, 0]], prompt_ids=[1, 7], drafter_rounds=[1])])
dv_rows = cd.per_prompt(dv_record)
check(dv_rows[1]["accepted"] == [0, 2, 1] and dv_rows[1]["prompt_ids"] == [1, 6] and dv_rows[0]["need"] == 4,
      "per prompt: each round's a_k and the prompt kept", dv_rows[1])
DRAFTER = {"slots": 32, "mask_token": 9, "width": 15, "tree": "beam", "beam": 8, "cov": True, "pool_r": 7,
           "topk": 16, "fill": 0.9, "causal": False, "temp": 1.0}
o_settings = {"forced": "ab" * 32, "model": "m", "canvas": None, "selector": None, "vlora": None, "drafter": DRAFTER,
              "max_new": 16, "eos": [2], "dtype": "float32"}
# The record's model file's DV: a beam of 31 with the oracle's other settings (flyx drafts it 15 wide); the oracle
# made from that very file.
DV_SECTION = {"slots": 32, "mask_token": 9, "tree": {"kind": "beam", "width": 31, "beam": 8, "cov": True}, "topk": 16,
              "fill": 0.9, "causal": False, "temp": 1.0, "source": {"canvas": "c.pt", "selector": "s.pt"}}
TARGET = {"dv": DV_SECTION, "vlora": False, "same_file": True}


def o_row(q, ids, prompt_ids, accepted, need, flyx_rounds, **kw):
    return dict({"q": q, "len": len(ids), "need": need, "flyx_rounds": flyx_rounds, "new": len(ids) - 1,
                 "ended": True, "accepted": accepted, "ids": ids, "prompt_ids": prompt_ids, "threads": 20}, **kw)


oracle = {0: o_row(0, [10, 11, 12, 13, 14, 15, 16, 17, 18, 2], [1, 5], [0, 3, 3, 0], 4, 3),
          1: o_row(1, [20, 21, 22, 23, 24, 2], [1, 6], [0, 1, 2], 3, 3)}
check(cd.drafter_name(DRAFTER) == "beam 15:8:cov, 32 slots, fill 0.9"
      and cd.drafter_name(dict(DRAFTER, tree="pool")) == "pool 15/7/1, 32 slots, fill 0.9", "oracle: its drafter named")


def compared(oracle_, settings=o_settings, record=dv_record, target=TARGET, min_same=None):
    return cd.compare_oracle(dv_rows, record, settings, oracle_, min_same, target)


# Forced to the text alone: the rounds up to the first difference, the forwards and TPF; not round k against round k.
lines, ok, numbers = compared(oracle)
check(ok and numbers["rounds_lockstep"] == 4 and numbers["rounds_compared"] == 6 and numbers["questions_alike"] == 1
      and "rounds_same" not in numbers, "oracle, text: the rounds before the first difference (the forward after "
      "flyx's last round not compared); the question whose every round agrees", numbers)
check(any("2 of 2 the ones it was forced to (1 of the record's questions not in the oracle)" in s for s in lines)
      and any("forwards (bifill-dv's): 7 here, 7 in the oracle; the same on 2 of 2" in s for s in lines)
      and any(s.endswith("forward 2 at token 1: a_k 2 / 1") for s in lines)
      and any(s.endswith("3/3       -") for s in lines) and "20 threads" in lines[0],
      "oracle, text: the report, where question 1's rounds part", lines)
check(abs(numbers["tpf_bifill"] - 16 / 7) < 1e-12 and abs(numbers["oracle_tpf_bifill"] - 16 / 7) < 1e-12
      and numbers["tpf_flyx"] == numbers["oracle_tpf_flyx"] == 2.0, "oracle, text: TPF both ways, both sides", numbers)
check(compared({**oracle, 0: dict(oracle[0], need=5)})[2]["questions_alike"] == 0,
      "oracle, text: a question whose rounds agree but whose forwards differ is not alike")
lines, ok, _ = compared(oracle, min_same=1)
check(not ok and any("needs an oracle that replayed" in s for s in lines),
      "oracle, text: --oracle-min refused (rounds after the first difference start elsewhere)", lines)

# Replayed (dv_forced.py --replay): every oracle forward starts where the record's round did; its own a_k compare.
r_settings = dict(o_settings, replay=True)
# Question 0: the oracle's tree would have accepted 6 drafts where the record's accepted 3, and 3 where it accepted 3
# (by its own counts it would have been clipped there: what replaying is for); question 1: every round the record's.
replayed = {0: dict(oracle[0], accepted=[0, 6, 3, 0], committed=[0, 3, 3, 0], replayed=3),
            1: dict(oracle[1], accepted=[0, 2, 1], committed=[0, 2, 1], replayed=3, need=3)}
lines, ok, numbers = compared(replayed, r_settings)
check(ok and numbers["rounds_same"] == 5 and numbers["rounds_compared"] == 6 and numbers["more"] == 0
      and numbers["fewer"] == 1 and numbers["drafts"] == 9 and numbers["oracle_drafts"] == 12
      and numbers["questions_alike"] == 1,
      "oracle, replayed: every round of the record against the oracle's forward there (its last own one not)", numbers)
check(any(s.endswith("forward 2 at token 1: a_k 3 / 6") for s in lines), "oracle, replayed: where they part", lines)
check(compared(replayed, r_settings, min_same=100.0 * 5 / 6)[1] and not compared(replayed, r_settings, min_same=84)[1],
      "oracle, replayed: --oracle-min fails below its share of rounds (5 of 6), not at it")
lines, ok, _ = compared({**replayed, 1: dict(replayed[1], committed=[0, 1, 2])}, r_settings)
check(not ok and "replayed rounds are not the record's (first: question 1)" in lines[-1],
      "oracle, replayed: replayed to other rounds than the record's: refused", lines)

# What ties the oracle to the record's DV and text: refused, nothing compared.
for why, kw, want in (
        ("a plain record", {"record": dict(dv_record, drafter="none", widths=[0])}, "drafter is 'none'"),
        ("rounds of two widths", {"record": dict(dv_record, widths=[15, 31])}, "draft [15, 31]"),
        ("rounds of another width", {"record": dict(dv_record, widths=[31])}, "width 15, the record's 31"),
        ("a pool tree's record", {"target": dict(TARGET, dv=dict(DV_SECTION, tree={"kind": "pool", "width": 31,
                                                                                    "ranks": 7}))},
         "tree 'beam', the record's 'pool'"),
        ("another beam", {"target": dict(TARGET, dv=dict(DV_SECTION, tree=dict(DV_SECTION["tree"], beam=4)))},
         "beam 8, the record's 4"),
        ("other slots", {"target": dict(TARGET, dv=dict(DV_SECTION, slots=64))}, "slots 32, the record's 64"),
        ("another fill", {"target": dict(TARGET, dv=dict(DV_SECTION, fill=0.5))}, "fill 0.9, the record's 0.5"),
        ("another MASK", {"target": dict(TARGET, dv=dict(DV_SECTION, mask_token=3))}, "mask_token 9"),
        ("a V-LoRA on one side", {"target": dict(TARGET, vlora=True)}, "V-LoRA"),
        ("another file", {"target": dict(TARGET, same_file=False)}, "another .flyx file"),
        ("a file that has no DV", {"target": dict(TARGET, dv=None)}, "has no DV"),
        ("a model file not read", {"target": {"error": "cannot be read"}}, "cannot be read"),
        ("answers cut elsewhere", {"settings": dict(o_settings, max_new=512)}, "cut at 512"),
        ("other end tokens", {"settings": dict(o_settings, eos=[2, 7])}, "end tokens [2, 7]")):
    lines, ok, _ = compared(oracle, **kw)
    check(not ok and want in lines[-1] and "nothing compared" in lines[-1], f"oracle: {why}: refused", lines)
lines, ok, _ = compared(oracle, target=dict(TARGET, same_file=None))
check(ok and any("warning: the oracle's weights are checkpoints" in s and "c.pt" in s for s in lines),
      "oracle: made from checkpoints, its drafter the record's: compared, the weights said to be untied", lines)
other_answer = {**oracle, 1: dict(oracle[1], ids=[20, 21, 22, 23, 25, 2])}
lines, ok, _ = compared(other_answer)
check(not ok and "question 1, token 4 of its answer" in lines[-1] and not any("rounds before" in s for s in lines),
      "oracle: forced to other answers: refused, nothing compared", lines)
lines, ok, _ = compared({**oracle, 0: dict(oracle[0], prompt_ids=[1])})
check(not ok and "question 0, its prompt" in lines[-1], "oracle: forced after another prompt: refused", lines)
lines, ok, _ = compared({5: oracle[0]})
check(not ok and "no question in common" in lines[-1], "oracle: no question in common fails", lines)

with tempfile.TemporaryDirectory() as tmp:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python", "convert"))
    from flyx_convert.fileformat import Writer

    def jsonl(name, rows_, end="\n", encoding="utf8"):
        path = os.path.join(tmp, name)
        with open(path, "w", encoding=encoding) as f:
            f.write("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows_) + end)
        return path

    def flyx_file(name, dv):
        """A .flyx file with only a header: the dv section of the drafter the record ran."""
        w = Writer()
        w.meta = {"arch": "llama", "dv": dv}
        path = os.path.join(tmp, name)
        w.write(path)
        return path
    model, other_model = flyx_file("m-dv.flyx", DV_SECTION), flyx_file("other-dv.flyx", dict(DV_SECTION, fill=0.8))
    copy = flyx_file("copy-dv.flyx", dict(DV_SECTION, source={"canvas": "other.pt"}))  # the same drafter, another file
    tied = dict(o_settings, model=cd.file_digest(model))
    o_path = jsonl("forced.jsonl", [{"settings": tied}] + list(oracle.values()))
    o_other = jsonl("other.jsonl", [{"settings": tied}] + list(other_answer.values()))
    r_path = jsonl("replayed.jsonl", [{"settings": dict(tied, replay=True)}] + list(replayed.values()))
    d_path, p_path = os.path.join(tmp, "dv.json"), os.path.join(tmp, "plain.json")
    with open(d_path, "w", encoding="utf8") as f:
        json.dump(dict(dv_record, model=model), f)
    # plain decoding's record of the same answers, beside it (gate 4): its rounds accept nothing
    with open(p_path, "w", encoding="utf8") as f:
        json.dump(dict(dv_record, model=model, drafter="none", widths=[0], prompts=[
            dict(p, trace=[[1, 0, 0]] * (len(p["ids"]) - 1), rounds=len(p["ids"]) - 1, drafter_rounds=[])
            for p in dv_record["prompts"]]), f)
    check(cd.read_oracle(o_path)[1][1]["accepted"] == [0, 1, 2], "oracle: read by question")
    bom = jsonl("bom.jsonl", [{"settings": tied}] + list(oracle.values()), encoding="utf-8-sig")
    check(cd.read_oracle(bom)[1][1]["accepted"] == [0, 1, 2], "oracle: a byte order mark is skipped")
    for name, rows_, end, want in (("cut.jsonl", [{"settings": tied}, oracle[0], json.dumps(oracle[1])[:40]], "",
                                    "line 3 is cut off"),
                                   ("junk.jsonl", [{"settings": tied}, "{junk", oracle[1]], "\n", "line 2 is not JSON"),
                                   ("free.jsonl", [{"settings": {"model": "m"}}], "\n", "not a teacher-forced")):
        try:
            cd.read_oracle(jsonl(name, rows_, end))
            check(False, f"oracle: {name} refused")
        except SystemExit as ex:
            check(want in str(ex.code), f"oracle: {name} refused", ex.code)
    check(main_quietly(["--drafted", d_path, "--oracle", o_path]) == 0
          and main_quietly(["--drafted", d_path, "--plain", p_path, "--oracle", r_path, "--oracle-min", "80"]) == 0,
          "check_dv --oracle: forced to the text, and replayed (--plain beside it: the drafted record's rounds)")
    check(main_quietly(["--drafted", d_path, "--oracle", o_path, "--oracle-min", "1"]) == 1
          and main_quietly(["--drafted", d_path, "--oracle", r_path, "--oracle-min", "90"]) == 1
          and main_quietly(["--drafted", d_path, "--oracle", o_other]) == 1,
          "check_dv --oracle: fails under --oracle-min, with --oracle-min and an oracle forced to the text alone, and "
          "on answers it was not forced to")
    check(main_quietly(["--drafted", d_path, "--oracle", o_path, "--record-model", other_model]) == 1
          and main_quietly(["--drafted", d_path, "--oracle", o_path, "--record-model", copy]) == 1
          and main_quietly(["--drafted", d_path, "--oracle", o_path, "--record-model",
                            os.path.join(tmp, "moved.flyx")]) == 1
          and main_quietly(["--drafted", p_path, "--oracle", o_path]) == 1,
          "check_dv --oracle: another model file (another DV; the same DV, another sha256), one it cannot read, a "
          "plain record: fail")

# ---- the files a check reads -------------------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    bom = os.path.join(tmp, "bom.jsonl")
    with open(bom, "w", encoding="utf-8-sig") as f:  # as PowerShell's Set-Content -Encoding utf8 writes it
        f.write(json.dumps({"question": "q", "answer": "#### 7"}) + "\n\n" + json.dumps({"answer": "#### 8"}) + "\n")
    check([r["answer"] for r in cd.read_jsonl(bom)] == ["#### 7", "#### 8"], "JSONL: a byte order mark, a blank line")
    check(cd.gold_answers(bom, 2) == ["7", "8"], "gold answers: read")
    for path, n, what in ((os.path.join(tmp, "none.jsonl"), 1, "a gold file that is not there"),
                          (None, 1, "no gold file at all"), (bom, 3, "fewer gold answers than prompts")):
        try:
            cd.gold_answers(path, n)
            check(False, f"gold answers: {what} refused")
        except SystemExit as ex:
            check(ex.code != 0, f"gold answers: {what} refused", ex.code)

    rec = dict(base, prompts=[prompt(0, [5, 2], [[1, 0, 0]])])
    rec_path = os.path.join(tmp, "plain.json")
    with open(rec_path, "w", encoding="utf8") as f:
        json.dump(rec, f)
    # before the tokenizer loads: --hf need not be a model here
    code = main_quietly(["--drafted", rec_path, "--hf", tmp, "--gsm8k", os.path.join(tmp, "typo.jsonl")])
    check(code not in (0, None), "check_dv: --hf with gold answers it cannot read fails", code)
    failed_path = os.path.join(tmp, "failed.json")
    with open(failed_path, "w", encoding="utf8") as f:
        json.dump(dict(rec, error="set.jsonl line 3: launch: device lost"), f)
    check(main_quietly(["--drafted", failed_path]) == 1 and main_quietly(["--drafted", rec_path]) == 0,
          "check_dv: a record of a run that stopped part-way fails")
    check(main_quietly(["--drafted", rec_path, "--plain", rec_path]) == 0, "check_dv: a plain record checked against "
          "itself passes")
    drafted_path = os.path.join(tmp, "drafted.json")
    with open(drafted_path, "w", encoding="utf8") as f:
        json.dump(dict(rec, prompts=[prompt(0, [5, 2], [[1, 0, 0]], drafted=3)]), f)
    check(main_quietly(["--drafted", rec_path, "--plain", drafted_path]) == 1, "check_dv: --plain a drafted run fails")
    # two answers of 4 tokens, in 2 forwards and in 1 (the end token accepted as a draft); the reference's the other
    # way round: the same TPF over both, half and twice it over each alone
    two = dict(base, prompts=[prompt(0, [5, 6, 7, 2], [[3, 2, 0]]), prompt(1, [5, 6, 7, 2], [[3, 2, 1]])])
    two_path, two_ref = os.path.join(tmp, "two.json"), os.path.join(tmp, "two_ref.json")
    with open(two_path, "w", encoding="utf8") as f:
        json.dump(two, f)
    with open(two_ref, "w", encoding="utf8") as f:
        json.dump({"source": "r.pt", "label": "x", "questions": {"0": {"len": 4, "need": 1},
                                                                  "1": {"len": 4, "need": 2}}}, f)
    check(main_quietly(["--drafted", two_path, "--ref", two_ref]) == 0
          and main_quietly(["--drafted", two_path, "--ref", two_ref, "--ranges", "0:2"]) == 0
          and main_quietly(["--drafted", two_path, "--ref", two_ref, "--ranges", "0:2,0:1"]) == 1,
          "check_dv --ref --ranges: the reference over each range's questions too (a range of them all once)")
    # DV's rounds chosen per round (plain decoding's, its warm-up, trees of 15 and 31) are not the
    # reference's, which drafts DV every round: --ref refuses the TPF (a pinned record of one width compares)
    dv_two = dict(two, drafter="dv", drafter_refused="", prompts=[dict(p, drafter_rounds=[1]) for p in two["prompts"]])
    chosen_path = os.path.join(tmp, "chosen.json")
    with open(chosen_path, "w", encoding="utf8") as f:
        json.dump(dict(dv_two, widths=[0, 15, 31, 0]), f)
    pinned_path = os.path.join(tmp, "pinned.json")
    with open(pinned_path, "w", encoding="utf8") as f:
        json.dump(dict(dv_two, widths=[31]), f)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cd.main(["--drafted", chosen_path, "--ref", two_ref])
    check(code == 1 and "TPF: not compared" in out.getvalue() and "--width N" in out.getvalue()
          and main_quietly(["--drafted", pinned_path, "--ref", two_ref]) == 0,
          "check_dv --ref: DV's rounds chosen among plain decoding's refused, DV pinned to one width compared",
          out.getvalue()[-400:])

print(f"{len(failures)} failed" if failures else "all passed")
sys.exit(1 if failures else 0)
