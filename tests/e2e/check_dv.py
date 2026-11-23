# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""A drafter on a prompt set against plain decoding: the same answers, tokens per forward, GSM8K accuracy.

Reads (or runs, then reads) two `flyx bench --prompts` records made with the same options but the drafter's: the
drafted run (DV, or the engine's default rounds) and plain decoding.

    # run both on GSM8K's test set as bifill-dv prompts it, then check (--limit: its first N questions); DV pinned
    # to its file's width, every round DV's as bifill-dv's references draft (--ref holds them to it)
    python tests/e2e/check_dv.py --flyx build/bin/flyx --model minicpm5-1b-dv-res2.flyx \\
        --prompts bifill-dv/data/gsm8k_test.jsonl --limit 100 --engine "--kv 16" --drafter "--drafter dv --width 31" \\
        --out-dir runs/dv --hf models/MiniCPM5-1B [--ref dyn_orig.json]
    # check two records already written (flyx bench MODEL --prompts ... --json FILE)
    python tests/e2e/check_dv.py --drafted dv.json --plain plain.json --hf models/MiniCPM5-1B [--ref REF.json]
    # DV's rounds against the reference's on the same text (dv_forced.py forced to plain.json's answers), and round
    # for round (dv_forced.py --replay dv.json: forced to its answers and rounds)
    python tests/e2e/check_dv.py --drafted dv.json --plain plain.json --oracle forced.jsonl
    python tests/e2e/check_dv.py --drafted dv.json --plain plain.json --oracle replayed.jsonl [--oracle-min 90]

Fails (exit 1) when
  * the two records differ in what they ran (model, K/V, limits, sampling, backend options, prompt tokens), or any
    prompt's generated tokens differ between them (gate 4: speculation never changes the output); --same-target:
    the plain record's model file may be another one of the same target (minicpm5-1b.flyx beside the DV file
    converted from the same checkpoint, minicpm5-1b-dv-res2.flyx), a warning;
  * a record says its run failed part-way ("error"), or the plain record drafted any token (not plain decoding);
  * the drafted record names a drafter that drafted no round, or says the backend refused rounds of its drafter
    ("drafter_refused": DV on a backend without its rounds decodes as without it); --refused-ok: the latter a
    warning (a DV file's own check against plain decoding);
  * --prompt-ids FILE: a prompt's tokens differ from that JSONL's "prompt" (bifill-dv's data/ar_test.jsonl holds
    the GSM8K prompts as transformers rendered them);
  * --hf: the gold answers cannot be read (--gsm8k, else the record's prompt set);
  * --ref REF.json: the drafted run's tokens per forward (bifill's count) differ from the reference's over the same
    questions by more than --tol percent (default: 1.5 from 1000 questions, 2 from 500, else 3); with --ranges, also
    over the questions of each range that holds fewer than all the run's, each at its own tolerance; refused (failed,
    TPF not compared) when the drafted record's DV did not draft every round at one width (rounds chosen beside plain
    decoding's: its plain and warm-up rounds are forwards bifill-dv's references do not have; --width N
    pins it).  A plain decoding reference (every answer's forwards its length: TPF 1) is compared for accuracy only.
  * --oracle FORCED.jsonl (python/reference/dv_forced.py: the reference forced to a record's answers), refused
    (nothing compared) when the drafted record's rounds are not the DV the oracle ran (a drafter other than dv, other
    than one width, a tree, width, slots, fill, MASK or V-LoRA other than its model file's DV drafting that width;
    another file, when the oracle was made from one: --record-model names the file where it is now), a question both
    hold has another prompt or answer than the oracle was forced to (its rounds ran on other text), the oracle cut
    answers or ended them otherwise than the record, no question is shared, or (replayed) the oracle replayed other
    rounds than the record's; --oracle-min PCT: an oracle replayed to the record's rounds (dv_forced.py --replay)
    whose a_k fewer than PCT percent of the record's rounds have; any oracle forced to the text alone.

Reports, for each run:
  * TPF as bifill-dv counts it, the primary number: an answer's length L = its tokens up to the first end token
    (included), at most --max-new; the forwards it needs = the fewest K whose first K rounds commit L tokens, where
    round k commits 1 + a_k (the token the round before chose, then the a_k drafts it accepted, an end token among
    them; DV's first round, all MASK, accepts none); R + 1 when R rounds never get there.  TPF = sum L / sum
    forwards (bifill-dv's evalgen.py and its mega engine, bifill/mega/engine.py).  Plain decoding gives exactly 1.
  * TPF as flyx bench counts it: sum (new tokens - 1) / sum rounds (new tokens without the end token).
  * mean L, ended answers, decode tok/s, the median prefill, and GSM8K accuracy as bifill-dv scores it (--hf: the
    tokenizer that decodes the answers; gold answers from --gsm8k, by default the record's prompt set): an answer
    counts only when it ended, its number taken by bifill-dv's gsm_pred (see BOXED below).
  * --ranges A:B,...: decode tok/s over prompts A .. B-1 of each record (a stretch of a run the machine was quiet).
  * against --ref: the reference's TPF over the same questions, how many questions need the same forwards,
    accuracy question by question, and (a reference with the answers' tokens: dv_ref_json.py --ids) how many
    answers are the reference's token for token; then the same over each --ranges range (the first 500 and 100 of
    1319: --ranges 0:500,0:100).  dv_ref_json.py writes REF.json from bifill-dv's result files.
  * against --oracle, over the questions both hold (each forward's a_k, the last one of an answer clipped to the
    tokens it still needs): replayed, every round against the oracle's forward that started where it did: the rounds
    with the oracle's a_k, those accepting fewer and more, the drafts accepted on both sides, the questions whose
    every round is the oracle's; forced to the text alone, the rounds before the first difference (after it the two
    start at other tokens with canvases slid by other counts), the forwards each needs and TPF both ways.  Per
    question, where the rounds first part (the forward, the answer token it was anchored at, both a_k).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

# bifill-dv's GSM8K prompt: the question, then this, as the one user message (its code/common.py and evalgen.py).
GSM8K_SUFFIX = "\nPlease reason step by step, and put your final answer within " + chr(92) + "boxed{}."
EOS = (1, 130073)  # MiniCPM5's end tokens, bifill-dv's EOS (a record names its model's: eos_ids)
MAX_NEW = 512      # bifill-dv's MAX_NEW

# What two runs to be compared must share (the record's fields; the drafter, lookup and widths may differ).
SAME = ("backend", "kv", "max_new_tokens", "ignore_eos", "suffix", "eos_ids", "sampling", "backend_options", "ctx")
# Command-line options that may differ between the drafted and the plain run, and how many values each takes
# (--json: its file, when one follows).
DRAFTER_FLAGS = {"--drafter": 1, "--drafter-file": 1, "--width": 1, "--no-spec": 0, "--no-lookup": 0,
                 "--draft-trees": 0}


# ---- counting ---------------------------------------------------------------------------------------------------

def answer_length(ids, eos=EOS, max_new=MAX_NEW):
    """L: the answer's tokens up to its first end token (included), at most max_new."""
    for k, t in enumerate(ids):
        if t in eos:
            return min(max_new, k + 1)
    return min(max_new, len(ids))


def ended(ids, eos=EOS):
    return any(t in eos for t in ids)


def forwards_needed(length, accepted):
    """bifill-dv's forwards for an answer of `length` tokens whose rounds accepted accepted[k] drafts each: the fewest
    K with K + sum(accepted[:K]) >= length (round k commits the token chosen before it, then its drafts), else
    len(accepted) + 1."""
    committed = 0
    for k, a in enumerate(accepted, 1):
        committed += 1 + a
        if committed >= length:
            return k
    return len(accepted) + 1


def accepted_by_round(trace):
    """bifill-dv's a_k from flyx bench's trace, a [tokens, accepted drafts, last token a draft] per round.  flyx's
    ring counts a draft that is a stop token among the round's tokens, not its accepted drafts (the round ends with
    it, as with the target's own token); bifill-dv counts it among the drafts accepted.  (Records written before the
    third value had two: read as 0.)"""
    return [step[1] + (step[2] if len(step) > 2 else 0) for step in trace]


def end_round_unknown(trace, length):
    """A record without the third value, whose answer's end token came in a round with other tokens: whether that
    end token was a draft (one forward fewer by bifill-dv's count) cannot be told."""
    emitted = 1  # the prefill's token
    for step in trace:
        emitted += step[0]
        if emitted >= length:
            return len(step) < 3 and step[0] > 1
    return False


def per_prompt(record):
    """Each prompt's L, forwards needed (bifill-dv's count) and tokens and rounds (flyx bench's), by prompt index."""
    eos = tuple(record.get("eos_ids", EOS))
    max_new = record.get("max_new_tokens", MAX_NEW)
    out = {}
    for p in record["prompts"]:
        L = answer_length(p["ids"], eos, max_new)
        done = ended(p["ids"], eos)
        accepted = accepted_by_round(p["trace"])
        out[p["i"]] = {"len": L, "need": forwards_needed(L, accepted), "ended": done, "accepted": accepted,
                       "unknown": done and end_round_unknown(p["trace"], L), "ids": p["ids"][:L],
                       "prompt_ids": p["prompt_ids"], "new": p["new_tokens"], "rounds": p["rounds"],
                       "decode_ms": p["decode_ms"], "prefill_ms": p["prefill_ms"]}
    return out


def tpf_bifill(rows):
    need = sum(r["need"] for r in rows)
    return sum(r["len"] for r in rows) / need if need else 0.0


def tpf_flyx(rows):
    rounds = sum(r["rounds"] for r in rows)
    return sum(r["new"] - 1 for r in rows) / rounds if rounds else 0.0


def decode_rate(rows):
    """Decode tok/s over the rows: their tokens after the first over their decode time."""
    ms = sum(r["decode_ms"] for r in rows)
    return sum(r["new"] - 1 for r in rows) * 1000.0 / ms if ms else 0.0


def median(values):
    v = sorted(values)
    return (v[(len(v) - 1) // 2] + v[len(v) // 2]) / 2 if v else 0.0


def speed_ranges(rows, ranges):
    """Decode tok/s over prompts a .. b-1 for each (a, b), with the median prompt's and the slowest's: lines."""
    lines = []
    for a, b in ranges:
        sel = [rows[i] for i in sorted(rows) if a <= i < b]
        rates = [(r["new"] - 1) * 1000.0 / r["decode_ms"] for r in sel if r["decode_ms"] > 0]
        lines.append(f"prompts {a}..{b - 1} ({len(sel)}): {decode_rate(sel):.1f} tok/s (median prompt "
                     f"{median(rates):.1f}, slowest {min(rates, default=0.0):.1f})")
    return lines


def parse_ranges(text):
    """"0:200,0:478" -> [(0, 200), (0, 478)]."""
    out = []
    for part in text.split(","):
        a, b = part.split(":")
        out.append((int(a), int(b)))
    return out


def default_tolerance(n):
    """Percent: what two counts of the same method's TPF were seen to differ by, by questions (plan, section 2)."""
    return 1.5 if n >= 1000 else 2.0 if n >= 500 else 3.0


# ---- GSM8K, scored as bifill-dv scores it ------------------------------------------------------------------------
# bifill-dv code/common.py gsm_gold / gsm_pred: the text after a "</think>", the last BOXED match, the last number in
# it (else in the text), "1,000" read as 1000, a trailing ".00" or ".0" dropped.  An answer counts only when it ended.
#
# BOXED is bifill-dv's pattern as it is written there, r"\boxed\{...": in a regular expression "\b" is a word
# boundary, not a backslash and a "b", so it finds "oxed{" after a character that is not a letter, digit or "_", and
# never LaTeX's "\boxed{".  Its answers' numbers are the last number of the whole text; kept so, for its counts.
BOXED = re.compile(r"\boxed\{([^}]*)\}")
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def gsm_gold(answer):
    return answer.split("####")[-1].strip().replace(",", "")


def gsm_pred(text):
    if "</think>" in text:
        text = text.split("</think>")[-1]
    boxed = BOXED.findall(text)
    numbers = NUMBER.findall(boxed[-1] if boxed else text)
    if not numbers:
        return None
    v = numbers[-1].replace(",", "")
    if v.endswith(".00"):
        return v[:-3]
    return v[:-2] if v.endswith(".0") else v


def gsm_correct(ids, text, gold, eos=EOS):
    return ended(ids, eos) and gsm_pred(text) == gold


def hf_decoder(hf_dir):
    """The answers' text as bifill-dv decodes them (transformers, special tokens skipped)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_dir)
    return lambda ids: tok.decode(ids, skip_special_tokens=True)


def read_jsonl(path):
    """A JSONL file's objects, blank lines skipped and a byte order mark too (as flyx bench --prompts reads a
    prompt set; PowerShell's utf8 writes one)."""
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def gold_answers(path, n):
    """The gold answers of a GSM8K JSONL ({answer: "... #### N"} in prompt order), at least n of them."""
    if not path or not os.path.exists(path):
        raise SystemExit(f"error: no gold answers to score against: {path!r} cannot be read (--gsm8k names the "
                         "file; the record's prompt set is a path relative to where flyx bench ran)")
    golds = [gsm_gold(r["answer"]) for r in read_jsonl(path)]
    if len(golds) < n:
        raise SystemExit(f"error: {path} has {len(golds)} gold answers; the records have prompt {n - 1}")
    return golds


def identical_answers(ours, theirs):
    """Answers (token lists by question) the same token for token over the questions both have:
    (identical, questions, mean length of the common prefix)."""
    common = [q for q in ours if q in theirs]
    same = sum(ours[q] == theirs[q] for q in common)
    prefix = sum(first_difference(ours[q], theirs[q]) for q in common)
    return same, len(common), prefix / len(common) if common else 0.0


# ---- the reference -----------------------------------------------------------------------------------------------

def ref_refusal(record):
    """Why a drafted record's TPF does not compare with a reference's (bifill-dv drafts DV every round), or None: its DV
    drafted rounds chosen among others (plain decoding's, its warm-up, trees of two widths), not every
    round at one width."""
    widths = record.get("widths") or []
    if record.get("drafter") == "dv" and len(widths) != 1:
        return (f"the record's DV rounds were chosen among {widths} (plain decoding, its warm-up and trees), not DV "
                "every round as the reference drafts: TPF compares with DV pinned (flyx --width N)")
    return None


def compare_ref(rows, ref, tol=None, refusal=None):
    """The drafted run's rows against REF.json over the questions both have: (report lines, passed).  refusal
    (ref_refusal): why the TPF does not compare (failed; accuracy still reported)."""
    per_q = ref["questions"]
    common = sorted(i for i in rows if str(i) in per_q)
    if not common:
        return ["  no question in common with the reference"], False
    ours = [rows[i] for i in common]
    theirs = [per_q[str(i)] for i in common]
    lines = [f"reference {ref.get('source')} [{ref.get('label')}], {len(common)} questions in common"]
    passed = True
    plain_ref = all(r["need"] == r["len"] for r in theirs)
    if plain_ref:
        lines.append("  plain decoding (TPF 1): accuracy only")
    elif refusal:
        lines.append(f"  TPF: not compared: {refusal} FAIL")
        passed = False
    else:
        tol = default_tolerance(len(common)) if tol is None else tol
        t_ours, t_ref = tpf_bifill(ours), tpf_bifill(theirs)
        diff = 100.0 * (t_ours / t_ref - 1.0)
        passed = abs(diff) <= tol
        lines.append(f"  TPF (bifill-dv's count): {t_ours:.4f} against {t_ref:.4f}: {diff:+.2f}% (tolerance "
                     f"{tol:g}%) {'PASS' if passed else 'FAIL'}")
        same_len = sum(a["len"] == b["len"] for a, b in zip(ours, theirs))
        same_need = sum(a["need"] == b["need"] for a, b in zip(ours, theirs))
        both = [(a, b) for a, b in zip(ours, theirs) if a["len"] == b["len"]]
        same_need_len = sum(a["need"] == b["need"] for a, b in both)
        lines.append(f"  forwards: {sum(a['need'] for a in ours)} against {sum(b['need'] for b in theirs)}; the same "
                     f"on {same_need} of {len(common)} questions ({same_need_len} of the {same_len} whose answers "
                     f"have the same length)")
    if all("correct" in r for r in ours):
        a = [r["correct"] for r in ours]
        b = [bool(r["correct"]) for r in theirs]
        lines.append(f"  correct: {sum(a)} against {sum(b)}; both {sum(x and y for x, y in zip(a, b))}, only here "
                     f"{sum(x and not y for x, y in zip(a, b))}, only in the reference "
                     f"{sum(y and not x for x, y in zip(a, b))}")
    if all("ids" in r for r in theirs) and all("ids" in r for r in ours):
        same, n, prefix = identical_answers({i: rows[i]["ids"] for i in common},
                                            {i: per_q[str(i)]["ids"] for i in common})
        lines.append(f"  answers identical token for token (up to the end token): {same} of {n}; mean common "
                     f"prefix {prefix:.1f} tokens")
    return lines, passed


def compare_ref_ranges(rows, ref, ranges, tol=None, refusal=None):
    """The same over the questions of each range A .. B-1 that holds fewer than all the rows (the whole run is
    compared already), each at its own tolerance (default_tolerance of its questions in common): a reference of all
    the questions held to the first N without cutting it.  [(range, report lines, passed)]."""
    out = []
    for a, b in ranges:
        part = {i: r for i, r in rows.items() if a <= i < b}
        if len(part) < len(rows):
            out.append(((a, b), *compare_ref(part, ref, tol, refusal)))
    return out


# ---- the teacher-forced oracle -----------------------------------------------------------------------------------
# python/reference/dv_forced.py runs the reference forced to a record's answers: the rounds DV runs on that text.
# A drafted record whose answers are those (gate 4: the plain record it was forced to) ran its rounds on the same
# text; on any other text they cannot compare.  Forced to the text alone, the oracle's rounds are the record's up to
# the first round whose a_k differs, and start at other tokens after it.  Replayed (dv_forced.py --replay on the
# drafted record itself), every oracle forward starts where the record's round did: every round compares.

def read_oracle(path):
    """dv_forced.py's results: (its settings, its rows by question).  Its first line holds the settings, the record it
    was forced to among them (`forced`, by sha256).  A byte order mark is skipped; a cut-off last line (a run stopped
    while writing it) or any line that is not JSON is refused."""
    with open(path, encoding="utf-8-sig") as f:
        text = f.read()
    lines = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            lines.append(json.loads(line))
        except ValueError as e:
            cut = not text.endswith("\n") and number == len(text.splitlines())
            raise SystemExit(f"error: {path} line {number} " + (
                "is cut off (a run stopped while writing it): dv_forced.py --out on it again completes it" if cut
                else f"is not JSON ({e})")) from None
    if not lines or "forced" not in lines[0].get("settings", {}):
        raise SystemExit(f"error: {path} is not a teacher-forced oracle's results (python/reference/dv_forced.py)")
    return lines[0]["settings"], {r["q"]: r for r in lines[1:]}


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def record_dv(path, settings):
    """What the drafted record's model file says of the DV it ran, to tie an oracle to it: {dv: its header's dv
    section, vlora: whether its target has adapters of its own, same_file: the oracle made from this very file
    (dv_forced.py --flyx: True or False; None when the oracle's weights were checkpoints)}, or {error: why not}."""
    if not path or not os.path.exists(path):
        return {"error": f"the record's model file {path!r} cannot be read, so the DV it ran is unknown "
                         "(--record-model names where it is now)"}
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python", "convert"))
    from flyx_convert.fileformat import Reader
    reader = Reader(path)
    header = reader.header
    reader.mm.close()
    reader.f.close()
    tensors = header.get("tensors", {})
    return {"dv": header.get("dv"),
            "vlora": any(n.startswith("target.") and n.endswith(".lora_a") for n in tensors),
            "same_file": file_digest(path) == settings.get("model") if settings.get("canvas") is None else None}


def drafter_differences(oracle, dv, width):
    """How the oracle's drafter (dv_forced.py's settings: a DvConfig) differs from the one a record ran: its file's dv
    section, `width` drafts a round (flyx drafts its rounds' width with the header's tree)."""
    tree = dv["tree"]
    want = {"tree": tree["kind"], "width": width, "slots": dv["slots"], "fill": dv["fill"],
            "mask_token": dv["mask_token"], "topk": dv["topk"], "causal": bool(dv["causal"]),
            "temp": dv.get("temp", 1.0)}
    if tree["kind"] == "beam":
        want.update(beam=tree["beam"], cov=bool(tree["cov"]))
    else:
        want["pool_r"] = tree["ranks"]
    return [f"{k} {oracle.get(k)!r}, the record's {v!r}" for k, v in want.items() if oracle.get(k) != v]


def answer_rounds(accepted, length):
    """The a_k of the forwards that commit an answer of `length` tokens, the last one's clipped to the tokens the
    answer still needs (a round may accept drafts past the end token or the length limit: they commit nothing)."""
    out, committed = [], 0
    for a in accepted:
        if committed >= length:
            break
        out.append(min(a, length - committed - 1))
        committed += 1 + out[-1]
    return out


def drafter_name(d):
    """A dv_forced.py settings' drafter, as the docs name it."""
    tree = (f"beam {d['width']}:{d['beam']}" + (":cov" if d["cov"] else "") if d["tree"] == "beam"
            else f"pool {d['width']}/{d['pool_r']}/1")
    return f"{tree}, {d['slots']} slots, fill {d['fill']:g}"


def oracle_question(row, orow, replayed):
    """One question's rounds against the oracle's: the record's a_k and the oracle's at the same forwards (replayed:
    its own a_k where it committed the record's counts; else its rounds, which after the first difference start at
    other tokens), and where they first part (the forward, the answer token it was anchored at, each side's a_k)."""
    a = answer_rounds(row["accepted"], row["len"])
    b = orow["accepted"][:orow["replayed"]] if replayed else answer_rounds(orow["accepted"], orow["len"])
    k = first_difference(a, b)
    return {"need": row["need"], "oracle_need": orow["need"], "compared": min(len(a), len(b)),
            "same": sum(x == y for x, y in zip(a, b)), "fewer": sum(x < y for x, y in zip(a, b)),
            "more": sum(x > y for x, y in zip(a, b)), "drafts": sum(a), "oracle_drafts": sum(b), "lockstep": k,
            "at": sum(1 + x for x in a[:k]), "here": a[k] if k < len(a) else None,
            "there": b[k] if k < len(b) else None}


def oracle_refusal(rows, record, settings, oracle, target):
    """Why the oracle's rounds do not compare with the record's (nothing is compared), or None: the record's rounds
    are not the DV the oracle ran (its drafter, width, tree, slots, fill, MASK, V-LoRA, its file when the oracle was
    made from one), the oracle's answers were cut or ended otherwise, no question is shared, a shared question's prompt
    or answer is not the one the oracle was forced to, or a replayed oracle replayed other rounds than the record's."""
    if record.get("drafter") != "dv":
        return f"the record's drafter is {record.get('drafter')!r}: its rounds are not DV's"
    widths = record.get("widths") or []
    if len(widths) != 1:
        return f"the record's rounds draft {widths}: DV's draft one width"
    if target is None or "error" in target:
        return (target or {}).get("error", "the record's model file was not read")
    if not target["dv"]:
        return "the record's model file has no DV"
    differ = drafter_differences(settings["drafter"], target["dv"], widths[0])
    if differ:
        return f"the oracle drafts otherwise than the record's DV: {'; '.join(differ)}"
    if bool(settings.get("vlora")) != target["vlora"]:
        return (f"the oracle's target {'has' if settings.get('vlora') else 'has no'} V-LoRA, the record's "
                f"{'has' if target['vlora'] else 'has none'}")
    if target["same_file"] is False:
        return "the oracle was made from another .flyx file than the record's model (sha256)"
    if settings.get("max_new") != record.get("max_new_tokens") or settings.get("eos") != record.get("eos_ids"):
        return (f"its answers were cut at {settings.get('max_new')} tokens and end tokens {settings.get('eos')}, the "
                f"record's at {record.get('max_new_tokens')} and {record.get('eos_ids')}")
    common = sorted(q for q in rows if q in oracle)
    if not common:
        return "no question in common with the drafted record"
    other = [q for q in common
             if rows[q]["ids"] != oracle[q]["ids"] or rows[q]["prompt_ids"] != oracle[q]["prompt_ids"]]
    if other:
        q = other[0]
        where = ("its prompt" if rows[q]["prompt_ids"] != oracle[q]["prompt_ids"] else
                 f"token {first_difference(rows[q]['ids'], oracle[q]['ids'])} of its answer")
        return (f"{len(other)} of {len(common)} questions' prompts or answers are not the ones the oracle was forced "
                f"to (first: question {q}, {where}); rounds on other text do not compare")
    if settings.get("replay"):
        other = [q for q in common if oracle[q]["committed"][:oracle[q]["replayed"]]
                 != answer_rounds(rows[q]["accepted"], rows[q]["len"])]
        if other:
            return (f"{len(other)} of {len(common)} questions' replayed rounds are not the record's (first: question "
                    f"{other[0]}): the oracle replayed another run")
    return None


def compare_oracle(rows, record, settings, oracle, min_same=None, target=None):
    """The drafted record's rounds against the oracle's over the questions both have: (report lines, passed, summary).
    target: record_dv() of the record's model file.  Refused (failed, nothing compared) on oracle_refusal's grounds.
    min_same: fail below this percentage of rounds with the oracle's a_k; only an oracle that replayed the record's
    rounds compares every round (forced to the text alone, rounds after the first difference start elsewhere)."""
    replayed = bool(settings.get("replay"))
    threads = sorted({r.get("threads") for r in oracle.values()}, key=str)
    lines = [f"oracle: teacher-forced reference ({drafter_name(settings['drafter'])}, {settings['dtype']}, "
             f"{', '.join(map(str, threads))} threads), forced to the record {settings['forced'][:12]}'s answers"
             + (" and rounds (replayed)" if replayed else "")]
    why = oracle_refusal(rows, record, settings, oracle, target)
    if why:
        return lines + [f"  FAIL: {why}; nothing compared"], False, {}
    if target["same_file"] is None:
        lines.append(f"  warning: the oracle's weights are checkpoints (dv_forced.py --model), the record's a file "
                     f"whose DV came from {target['dv'].get('source')}: its drafter is the record's, its weights are "
                     "not tied to them")
    if len(threads) > 1:
        lines.append(f"  warning: the oracle's rows were made at {len(threads)} thread counts")
    common = sorted(q for q in rows if q in oracle)
    per = {q: oracle_question(rows[q], oracle[q], replayed) for q in common}
    total = {k: sum(p[k] for p in per.values()) for k in ("same", "compared", "fewer", "more", "drafts",
                                                          "oracle_drafts", "lockstep")}
    share = 100.0 * total["same"] / total["compared"] if total["compared"] else 0.0
    ours, theirs = [rows[q] for q in common], [oracle[q] for q in common]
    lines.append(f"  answers: {len(common)} of {len(common)} the ones it was forced to ({len(rows) - len(common)} of "
                 f"the record's questions not in the oracle)")
    if replayed:
        alike = sum(p["same"] == p["compared"] for p in per.values())
        passed = min_same is None or share >= min_same
        lines += [f"  rounds with the oracle's a_k (each oracle forward where the record's round started): "
                  f"{total['same']} of {total['compared']} ({share:.2f}%)"
                  + ("" if min_same is None else f" (at least {min_same:g}%) {'PASS' if passed else 'FAIL'}"),
                  f"  rounds accepting fewer drafts here {total['fewer']}, more {total['more']}; drafts accepted "
                  f"{total['drafts']} here, {total['oracle_drafts']} by the oracle on the same rounds "
                  f"({100.0 * (total['drafts'] / max(total['oracle_drafts'], 1) - 1):+.2f}%)",
                  f"  questions whose every round is the oracle's: {alike} of {len(common)}"]
        t_there = f_there = None
    else:
        alike = sum(p["lockstep"] == p["compared"] and p["need"] == p["oracle_need"] for p in per.values())
        passed = min_same is None
        t_there = sum(r["len"] for r in theirs) / sum(r["need"] for r in theirs)
        f_there = sum(r["new"] - 1 for r in theirs) / max(1, sum(r["flyx_rounds"] for r in theirs))
        t_here = tpf_bifill(ours)
        lines += [f"  rounds before the first difference: {total['lockstep']} of {total['compared']}; questions whose "
                  f"every round is the oracle's: {alike} of {len(common)} (after a difference the two start at other "
                  f"tokens, so only totals compare)",
                  f"  forwards (bifill-dv's): {sum(r['need'] for r in ours)} here, {sum(r['need'] for r in theirs)} in "
                  f"the oracle; the same on {sum(p['need'] == p['oracle_need'] for p in per.values())} of "
                  f"{len(common)} questions",
                  f"  TPF, bifill-dv's: {t_here:.4f} here, {t_there:.4f} in the oracle "
                  f"({100.0 * (t_here / t_there - 1):+.2f}%)",
                  f"  TPF, flyx bench's: {tpf_flyx(ours):.4f} here, {f_there:.4f} in the oracle"]
        if min_same is not None:
            lines.append("  FAIL: --oracle-min holds every round to the oracle's, which needs an oracle that replayed "
                         "this record's rounds (dv_forced.py --replay): forced to the text alone, its rounds after "
                         "the first difference start at other tokens")
    lines.append(f"  {'q':>5} {'L':>4} {'need':>9} {'rounds same':>12}  first difference (here / oracle)")
    for q in common:
        p = per[q]
        part = ("-" if p["lockstep"] == p["compared"] else
                f"forward {p['lockstep'] + 1} at token {p['at']}: a_k {p['here']} / {p['there']}")
        lines.append(f"  {q:5d} {rows[q]['len']:4d} {p['need']:4d}/{p['oracle_need']:<4d} "
                     f"{p['same']:5d}/{p['compared']:<6d}  {part}")
    summary = {"replayed": replayed, "questions": len(common), "questions_alike": alike, "rounds_compared":
               total["compared"], "rounds_lockstep": total["lockstep"], "tpf_bifill": tpf_bifill(ours),
               "tpf_flyx": tpf_flyx(ours), "per_question": {str(q): per[q] for q in common}}
    if replayed:
        summary.update(rounds_same=total["same"], fewer=total["fewer"], more=total["more"], drafts=total["drafts"],
                       oracle_drafts=total["oracle_drafts"])
    else:
        summary.update(oracle_tpf_bifill=t_there, oracle_tpf_flyx=f_there)
    return lines, passed, summary


# ---- the two records ---------------------------------------------------------------------------------------------

def options_without_drafter(record):
    """The record's command line after `flyx`, without the options the drafted and the plain run differ by."""
    args = record.get("command", [])[1:]
    out, k = [], 0
    while k < len(args):
        if args[k] in DRAFTER_FLAGS:
            k += 1 + DRAFTER_FLAGS[args[k]]
        elif args[k] == "--json":
            k += 2 if k + 1 < len(args) and not args[k + 1].startswith("-") else 1
        else:
            out.append(args[k])
            k += 1
    return out


def compare_records(drafted, plain, same_target=False):
    """What the two records ran differs: (failure lines, warning lines).  same_target: the model files may differ
    (another file of the same target), a warning."""
    fails, warns = [], []
    for key in SAME:
        if drafted.get(key) != plain.get(key):
            fails.append(f"'{key}' differs: {drafted.get(key)!r} against {plain.get(key)!r}")
    if os.path.basename(drafted["model"]) != os.path.basename(plain["model"]):
        (warns if same_target else fails).append(f"models differ: {drafted['model']} against {plain['model']}")
    drafts = sum(p["drafted"] for p in plain["prompts"])
    if drafts:
        fails.append(f"the plain record is not plain decoding: its rounds drafted {drafts} tokens (drafter "
                     f"{plain.get('drafter')!r}, lookup {plain.get('lookup')!r})")
    pd = {p["i"]: p for p in drafted["prompts"]}
    pp = {p["i"]: p for p in plain["prompts"]}
    if sorted(pd) != sorted(pp):
        fails.append(f"prompt sets differ: {len(pd)} prompts against {len(pp)}")
    else:
        differ = [i for i in pd if pd[i]["prompt_ids"] != pp[i]["prompt_ids"]]
        if differ:
            fails.append(f"{len(differ)} prompts' tokens differ (first: prompt {differ[0]})")
    ours, theirs = options_without_drafter(drafted), options_without_drafter(plain)
    if same_target:  # the model files are said above
        ours, theirs = [a for a in ours if a != drafted["model"]], [a for a in theirs if a != plain["model"]]
    if ours != theirs:
        warns.append(f"command lines differ beyond the drafter's options: {ours} against {theirs}")
    return fails, warns


def drafter_ran(record, refused_ok=False):
    """Whether the drafted record's drafter drafted: (failure lines, warning lines).  A record that names a drafter
    whose rounds never ran (records before flyx said so named a refused one), or whose drafter the backend refused at
    some width or at all ("drafter_refused": the rounds then are the ones without it), would hold plain decoding or
    lookup to the drafter's numbers; refused_ok: a refusal is a warning (a file's own check against plain decoding)."""
    fails, warns = [], []
    name = record.get("drafter")
    rounds = sum(sum(p.get("drafter_rounds") or []) for p in record["prompts"])
    if name not in (None, "none") and rounds == 0:
        fails.append(f"the drafted record's drafter {name!r} drafted no round (its rounds: {record.get('rounds')})")
    if record.get("drafter_refused"):
        (warns if refused_ok else fails).append(
            f"the backend refused rounds of the drafter ({record['drafter_refused']}); the record's drafter: "
            f"{name!r}, its rounds {record.get('rounds')}, lookup {record.get('lookup')!r}")
    return fails, warns


def first_difference(a, b):
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k
    return min(len(a), len(b))


def summarize(name, record, rows):
    vals = list(rows.values())
    out = {"run": name, "prompts": len(vals), "tpf_bifill": tpf_bifill(vals), "tpf_flyx": tpf_flyx(vals),
           "mean_len": sum(r["len"] for r in vals) / len(vals), "ended": sum(r["ended"] for r in vals),
           "forwards": sum(r["need"] for r in vals), "rounds": sum(r["rounds"] for r in vals),
           "decode_tok_s": decode_rate(vals), "prefill_ms_median": median([r["prefill_ms"] for r in vals]),
           "drafter": record.get("drafter"), "lookup": record.get("lookup")}
    if all("correct" in r for r in vals):
        out["correct"] = sum(r["correct"] for r in vals)
    return out


def run_bench(args, extra, out):
    # The executable's full path: Windows does not run a relative one written with forward slashes.
    cmd = [os.path.abspath(args.flyx), "bench", args.model, "--prompts", args.prompts, "--suffix", args.suffix]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    cmd += args.engine.split() + extra.split() + ["--json", out]
    print("running:", " ".join(cmd), flush=True)
    if subprocess.run(cmd).returncode != 0:
        sys.exit(f"error: flyx bench failed: {' '.join(cmd)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--drafted", help="the drafted run's record (written there when --flyx runs it)")
    ap.add_argument("--plain", help="plain decoding's record, the same options otherwise (optional when reading)")
    ap.add_argument("--same-target", action="store_true",
                    help="the plain record's model may be another file of the same target (a warning, not a failure)")
    ap.add_argument("--refused-ok", action="store_true",
                    help="a drafter the backend refused (the drafted run decoding as without it) is a warning, not a "
                         "failure: a file's own check against plain decoding")
    run = ap.add_argument_group("running both (else the records are read)")
    run.add_argument("--flyx", help="the flyx executable")
    run.add_argument("--model")
    run.add_argument("--prompts", help="the prompt set (flyx bench --prompts), e.g. bifill-dv's data/gsm8k_test.jsonl")
    run.add_argument("--suffix", default=GSM8K_SUFFIX.replace("\n", "\\n"),
                     help="after each question (flyx bench --suffix; default: bifill-dv's GSM8K instruction)")
    run.add_argument("--limit", type=int, default=0, help="the first N prompts")
    run.add_argument("--engine", default="", help="options of both runs, e.g. \"--kv 16 --backend-opt K=V\"")
    run.add_argument("--drafter", default="", help="the drafted run's own options (default: the engine's defaults)")
    run.add_argument("--plain-options", default="--no-spec --no-lookup", help="plain decoding's own options")
    run.add_argument("--out-dir", default=".", help="where the records go (drafted.json, plain.json)")
    ap.add_argument("--hf", help="the model's Hugging Face directory: its tokenizer decodes answers for accuracy")
    ap.add_argument("--gsm8k", help="gold answers, a JSONL of {answer: '... #### N'} in prompt order (default: the "
                                    "drafted record's prompt set)")
    ap.add_argument("--ref", help="REF.json (dv_ref_json.py) to hold the drafted run's TPF to")
    ap.add_argument("--tol", type=float, help="percent the TPF may differ from --ref's (default: by questions)")
    ap.add_argument("--oracle", help="python/reference/dv_forced.py's results, forced to the drafted record's answers "
                                     "(--replay: and its rounds): its rounds compared with the drafted record's")
    ap.add_argument("--oracle-min", type=float,
                    help="percent of the rounds that must have the oracle's a_k, an oracle that replayed the record's "
                         "rounds (default: report only)")
    ap.add_argument("--record-model", help="the drafted record's model file, where it is now (default: the path the "
                                           "record names), whose DV the oracle's is tied to")
    ap.add_argument("--prompt-ids", help="a JSONL whose 'prompt' (token ids; by 'i', else line order) every prompt's "
                                         "tokens must equal, e.g. bifill-dv's data/ar_test.jsonl")
    ap.add_argument("--ranges", type=parse_ranges, default=[],
                    help="decode tok/s over these ranges of each record's prompts, A:B (A .. B-1), comma-separated; "
                         "with --ref, the reference over each range's questions too")
    ap.add_argument("--summary", help="write the numbers as JSON here")
    args = ap.parse_args(argv)

    if args.flyx:
        if not (args.model and args.prompts):
            ap.error("--flyx needs --model and --prompts")
        args.drafted = args.drafted or os.path.join(args.out_dir, "drafted.json")
        args.plain = args.plain or os.path.join(args.out_dir, "plain.json")
        for path in (args.drafted, args.plain):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        run_bench(args, args.drafter, args.drafted)
        run_bench(args, args.plain_options, args.plain)
    if not args.drafted:
        ap.error("--drafted (a record to read) or --flyx (to run both) is needed")

    records = {"drafted": read_json(args.drafted)}
    if args.plain:
        records["plain"] = read_json(args.plain)
    rows = {name: per_prompt(rec) for name, rec in records.items()}
    failed = False

    for name, rec in records.items():
        print(f"{name}: {getattr(args, name)}\n  {' '.join(rec.get('command', []))}")
        if "error" in rec:
            print(f"FAIL: the run stopped after {len(rec['prompts'])} prompts: {rec['error']}")
            failed = True
        unknown = sum(r["unknown"] for r in rows[name].values())
        if unknown:
            print(f"warning: {unknown} answers' end tokens came in a round with other tokens, and the record (written "
                  "before its trace said whether a round's last token was a draft) cannot tell whether they were "
                  "drafts: counted as the rounds' own tokens (a forward more each by bifill-dv's count, if drafts)")

    fails, warns = drafter_ran(records["drafted"], args.refused_ok)
    for w in warns:
        print("warning:", w)
    for f in fails:
        print("FAIL:", f)
    failed |= bool(fails)
    if "plain" in records:
        fails, warns = compare_records(records["drafted"], records["plain"], args.same_target)
        for w in warns:
            print("warning:", w)
        for f in fails:
            print("FAIL:", f)
        failed |= bool(fails)
        pd = {p["i"]: p["ids"] for p in records["drafted"]["prompts"]}
        pp = {p["i"]: p["ids"] for p in records["plain"]["prompts"]}
        differ = [i for i in sorted(pd) if i in pp and pd[i] != pp[i]]
        print(f"answers: {len(pd) - len(differ)} of {len(pd)} identical token for token to plain decoding "
              f"{'PASS' if not differ else 'FAIL'}")
        for i in differ[:10]:
            print(f"  prompt {i}: first difference at token {first_difference(pd[i], pp[i])} "
                  f"({len(pd[i])} tokens against {len(pp[i])})")
        failed |= bool(differ)

    if args.prompt_ids:
        want = {r.get("i", k): r["prompt"] for k, r in enumerate(read_jsonl(args.prompt_ids))}
        for name, rec in records.items():
            bad = [p["i"] for p in rec["prompts"] if want.get(p["i"]) != p["prompt_ids"]]
            print(f"prompt tokens ({name}): {len(rec['prompts']) - len(bad)} of {len(rec['prompts'])} equal to "
                  f"{os.path.basename(args.prompt_ids)}'s {'PASS' if not bad else 'FAIL (first: prompt %d)' % bad[0]}")
            failed |= bool(bad)

    if args.hf:
        golds = gold_answers(args.gsm8k or records["drafted"].get("prompt_set"),
                             1 + max(p["i"] for rec in records.values() for p in rec["prompts"]))
        decode = hf_decoder(args.hf)
        for name, rec in records.items():
            eos = tuple(rec.get("eos_ids", EOS))
            for p in rec["prompts"]:
                rows[name][p["i"]]["correct"] = gsm_correct(p["ids"], decode(p["ids"]), golds[p["i"]], eos)
    else:
        print("accuracy: not scored (--hf names the tokenizer that decodes the answers)")

    summary = {name: summarize(name, records[name], rows[name]) for name in records}
    names = list(records)
    print(f"\n{'':34s}" + "".join(f"{n:>14s}" for n in names))
    for label, key, fmt in (("TPF, bifill-dv's (sum L / forwards)", "tpf_bifill", "{:14.4f}"),
                            ("TPF, flyx bench's ((new-1)/rounds)", "tpf_flyx", "{:14.4f}"),
                            ("forwards (bifill-dv's)", "forwards", "{:14d}"), ("rounds", "rounds", "{:14d}"),
                            ("mean length L", "mean_len", "{:14.2f}"), ("ended", "ended", "{:14d}"),
                            ("decode tok/s", "decode_tok_s", "{:14.1f}"),
                            ("prefill ms, median prompt", "prefill_ms_median", "{:14.1f}"),
                            ("GSM8K correct", "correct", "{:14d}")):
        if all(key in summary[n] for n in names):
            print(f"{label:34s}" + "".join(fmt.format(summary[n][key]) for n in names))
    if all("correct" in summary[n] for n in names):
        print(f"{'GSM8K accuracy':34s}" + "".join(f"{100.0 * summary[n]['correct'] / summary[n]['prompts']:13.2f}%"
                                                  for n in names))

    for name in names if args.ranges else ():
        lines = speed_ranges(rows[name], args.ranges)
        summary[name]["ranges"] = lines
        print(f"\ndecode speed by prompt range, {name}:\n  " + "\n  ".join(lines))

    if args.ref:
        ref = read_json(args.ref)
        refusal = ref_refusal(records["drafted"])
        lines, passed = compare_ref(rows["drafted"], ref, args.tol, refusal)
        print("\n" + "\n".join(lines))
        summary["ref"] = {"path": args.ref, "passed": passed, "report": lines, "ranges": []}
        for (a, b), part, ok in compare_ref_ranges(rows["drafted"], ref, args.ranges, args.tol, refusal):
            print(f"\nprompts {a}..{b - 1}: " + "\n".join(part))
            summary["ref"]["ranges"].append({"range": [a, b], "passed": ok, "report": part})
            passed &= ok
        summary["ref"]["passed"] = passed
        failed |= not passed

    if args.oracle:
        settings, oracle = read_oracle(args.oracle)
        drafted = records["drafted"]
        target = (record_dv(args.record_model or drafted.get("model"), settings) if drafted.get("drafter") == "dv"
                  else None)  # refused on its drafter before a model file is read
        lines, passed, numbers = compare_oracle(rows["drafted"], drafted, settings, oracle, args.oracle_min, target)
        print("\n" + "\n".join(lines))
        summary["oracle"] = dict(numbers, path=args.oracle, passed=passed)
        failed |= not passed

    if args.summary:
        with open(args.summary, "w", encoding="utf8") as f:
            json.dump(summary, f, indent=1)
    print("\nFAIL" if failed else "\nPASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
