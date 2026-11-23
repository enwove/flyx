# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The DV reference (flyx_ref/dv.py) teacher-forced to the answers of a flyx bench record, and with --replay to its
rounds too, so that flyx's DV rounds can be held to the reference's on the same text.

    python -I python/reference/dv_forced.py RECORD.json --model MODEL_DIR --canvas CANVAS.pt \
        --selector SELECTOR.pt [--vlora VLORA.pt] [--mask-token TOKEN] [--tree beam:31:8:cov] [--slots 32] \
        [--fill 0.9] --out FORCED.jsonl
    python -I python/reference/dv_forced.py RECORD.json --flyx MODEL-DV.flyx [--tree pool:15:7:1] --out FORCED.jsonl
        [--replay] [--first 0] [--count 50] [--dtype float32] [--threads N] [--free FREE.jsonl]

Why forcing: a drafter never changes the output (AGENTS.md gate 4), so flyx's DV rounds see the text its plain
decoding wrote.  The reference's own float32 greedy answers part from flyx's at near-ties, and rounds on two texts
cannot be compared; forced to the record's answers (``Dv.generate(forced=...)``: every draft is checked against the
record's next token instead of the target's choice) it drafts from the text flyx's rounds draft from.  Its rounds and
flyx's then agree up to the first round whose a_k differs (other rounding turns a tree at a near-tie); after it, the
two start at other tokens with canvases slid by other counts, and only totals (TPF) compare.

--replay (RECORD a drafted run's record): every forward also commits the drafts the record's round committed
(``Dv.generate(commits=...)``), whatever the reference's own tree accepted, so that each of its forwards starts where
the record's round did, at its anchor, with its canvas slid by the record's counts.  Its own a_k (the drafts its tree
would have accepted there) then compare with the record's round for round, every round: `check_dv.py --oracle`.

RECORD: `flyx bench --prompts ... --json`, plain decoding's or a drafted run's (whose answers are plain decoding's):
per prompt its index `i`, `prompt_ids`, `ids` and `trace`; the run's `eos_ids`, `max_new_tokens` and `drafter`.
Question i is forced to its answer's first L tokens (up to the first end token, included, at most max_new_tokens:
tests/e2e/check_dv.py's L) after its prompt as the record holds it.  The DV assets are bifill-dv's checkpoints beside
the target's Hugging Face directory (--model, --canvas, --selector; the tree, slots and fill as dv_gsm8k.py takes
them; --mask-token the MASK token's text, MiniCPM5's by default), or a .flyx file with DV (--flyx: the target, canvas
and selector as stored; its header's tree, slots and fill unless given).

Per question, one JSON line: q, `len` (L), the answer and prompt it was forced to (`ids`, `prompt_ids`: check_dv.py
--oracle holds a drafted record to them), `accepted` (a_k of every forward, the first all-MASK one included, up to the
one that commits the answer's last token), with --replay `committed` (the drafts each forward committed) and
`replayed` (how many forwards committed the record's counts; a last one past the record's rounds commits its own),
`need` (bifill's forwards) and `flyx_rounds` and `new` (flyx bench's count) of what it committed, `secs` and
`threads` (float32 products round differently with another thread count, which can turn a tree at a near-tie).
Totals: TPF both ways.  --out keeps the run's settings in its first line as dv_gsm8k.py's does (the record by its
sha256 under `forced`, `replay` when on) and goes on only with the same settings, at the same thread count.

--free FREE.jsonl: a dv_gsm8k.py run of the same model, canvas, selector, drafter, cap and dtype (and each question
at the same thread count, where the free run kept it).  On the questions whose free answer is the record's, the
forced rounds must be the free run's, a_k for a_k (``answer_rounds``: a forward's drafts past the answer's last token
commit nothing); exit 1 otherwise.  Not with --replay.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dv_gsm8k import (MASK_TOKEN, differences, drafter, file_digest, model_digest, open_results,  # noqa: E402
                      read_results, threads_refused, tree_name)
from flyx_ref.dv import (Dv, canvas_from_pt, from_flyx, hf_llama, selector_from_pt,  # noqa: E402
                         vlora_from_pt)

# What a free run must share with a forced one for their rounds to be compared (the free run's questions and the
# forced run's record are what differ).
SHARED = ("model", "canvas", "selector", "vlora", "drafter", "max_new", "dtype")


def arguments():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("record", help="flyx bench --prompts --json record whose answers the reference is forced to")
    assets = ap.add_mutually_exclusive_group(required=True)
    assets.add_argument("--model", help="the target's Hugging Face directory (with --canvas and --selector)")
    assets.add_argument("--flyx", help="a .flyx file with DV: the target, the canvas and the selector")
    ap.add_argument("--canvas")
    ap.add_argument("--selector")
    ap.add_argument("--vlora")
    ap.add_argument("--mask-token", default=MASK_TOKEN,
                    help="with --model: the MASK token's text (default %(default)s)")
    ap.add_argument("--tree", help="beam:T:W[:cov] or pool:T:R[:1] (default: beam:31:8:cov, or the .flyx file's)")
    ap.add_argument("--slots", type=int, default=0, help="canvas slots (0: the canvas's own, or the .flyx file's)")
    ap.add_argument("--fill", type=float, help="the canvas fill threshold (default: 0.9, or the .flyx file's)")
    ap.add_argument("--replay", action="store_true",
                    help="commit the record's own counts each forward (RECORD a drafted run's): every round compares")
    ap.add_argument("--first", type=int, default=0, help="the first question (the record's prompt index)")
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--dtype", default="float32", choices=("float64", "float32", "bfloat16"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--free", help="a dv_gsm8k.py results file of the same drafter, checked where its answers agree")
    ap.add_argument("--threads", type=int, default=0)
    return ap


def answer_length(ids, eos, cap):
    """L: the answer's tokens up to its first end token (included), at most cap."""
    for k, t in enumerate(ids):
        if t in eos:
            return min(cap, k + 1)
    return min(cap, len(ids))


def record_rounds(trace):
    """A flyx bench trace's a_k as bifill-dv counts them (check_dv.py's accepted_by_round): each round's accepted
    drafts, an end token accepted as a draft among them."""
    return [step[1] + (step[2] if len(step) > 2 else 0) for step in trace]


def answer_rounds(accepted, length):
    """The a_k of the forwards that commit an answer of `length` tokens, the last one's clipped to the tokens the
    answer still needs: a free run's last forward may accept drafts past the end token or the cap, which commit
    nothing (a forced run never does: there is no record token to check them against)."""
    out, committed = [], 0
    for a in accepted:
        if committed >= length:
            break
        out.append(min(a, length - committed - 1))
        committed += 1 + out[-1]
    return out


def read_record(path, replay=False):
    """A flyx bench record -> (its end tokens, its length cap, {i: {prompt_ids, ids: the answer's first L; replay:
    commits, the drafts each of its rounds committed to them}})."""
    with open(path, encoding="utf-8-sig") as f:
        rec = json.load(f)
    if replay and rec.get("drafter") in (None, "none"):
        raise ValueError(f"--replay: {path} is a run without a drafter ({rec.get('drafter')!r}): no rounds to replay")
    eos, cap = tuple(rec["eos_ids"]), rec["max_new_tokens"]
    answers = {}
    for p in rec["prompts"]:
        L = answer_length(p["ids"], eos, cap)
        answers[p["i"]] = {"prompt_ids": p["prompt_ids"], "ids": p["ids"][:L]}
        if replay:
            answers[p["i"]]["commits"] = answer_rounds(record_rounds(p["trace"]), L)
    return eos, cap, answers


def drafter_config(a, dv_header, canvas_cfg, mask_token):
    """The DvConfig of the run: a .flyx file's header with the options given over it, else dv_gsm8k.py's from the
    checkpoint's canvas (its defaults: beam:31:8:cov, fill 0.9)."""
    if dv_header is None:
        return drafter(argparse.Namespace(tree=a.tree or "beam:31:8:cov", slots=a.slots,
                                          fill=0.9 if a.fill is None else a.fill), canvas_cfg, mask_token)
    dv = dv_header
    if a.tree:
        dv.set_tree(a.tree)
    if a.slots:
        dv.slots = a.slots
    if a.fill is not None:
        dv.fill = a.fill
    return dv


def flyx_vlora(path):
    """The V-LoRA a .flyx file's target holds (python/convert --lora, or beside its DV): its source's sha256 as the
    header's lora section names it; None without one (or for a file that is no .flyx file, which assets() refuses)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "convert"))
    from flyx_convert.fileformat import Reader
    try:
        r = Reader(path)
    except ValueError:
        return None
    lora = r.header.get("lora")
    r.mm.close()
    r.f.close()
    return ((lora.get("source") or {}).get("sha256") or "in the file") if lora else None


def run_settings(a, dv, eos, cap):
    """What decides the run's rounds, as --out keeps it: the record (by its sha256), the target and DV's weights (the
    V-LoRA: --vlora's checkpoint, or the one a --flyx file holds), the drafter, the cap and end tokens, the dtype,
    whether the record's rounds are replayed.  --first, --count, --free and --out are not in it; the thread count is
    kept in each row."""
    digest = lambda p: file_digest(p) if p else None  # noqa: E731
    vlora = digest(a.vlora) if not a.flyx else flyx_vlora(a.flyx)
    settings = {"forced": file_digest(a.record), "model": digest(a.flyx) if a.flyx else model_digest(a.model),
                "canvas": digest(a.canvas), "selector": digest(a.selector), "vlora": vlora,
                "drafter": dataclasses.asdict(dv), "max_new": cap, "eos": list(eos), "dtype": a.dtype}
    if a.replay:
        settings["replay"] = True
    return settings


def forced_row(q, answer, g, eos, secs, threads):
    """One question's line from Dv.generate's result, forced to `answer` (and its rounds' commits, if replayed)."""
    ids = answer["ids"]
    if g["ids"] != ids:
        raise RuntimeError(f"question {q}: the forced run did not emit the record's answer ({len(g['ids'])} tokens "
                           f"against {len(ids)})")
    commits = answer.get("commits")
    if commits is not None and g["committed"][:len(commits)] != commits:
        raise RuntimeError(f"question {q}: the replayed run did not commit the record's rounds")
    ended = bool(ids) and ids[-1] in eos
    row = {"q": q, "len": len(ids), "need": g["need"], "flyx_rounds": g["flyx_rounds"], "new": len(ids) - ended,
           "ended": ended, "accepted": g["accepted"], "ids": ids, "prompt_ids": answer["prompt_ids"], "secs": secs,
           "threads": threads}
    if commits is not None:
        row["committed"], row["replayed"] = g["committed"], len(commits)
    return row


def force(model, answers, qs, eos, cap, done, out, threads):
    """Each question of qs forced to its answer (a Dv, or anything with its generate), its line written to `out`;
    the questions `done` holds are taken from it.  Returns the rows in qs's order."""
    rows = []
    print(f"{'q':>5} {'L':>4} {'need':>4} {'flyx':>4} {'TPF':>6} {'secs':>6}", flush=True)
    for q in qs:
        if q in done:
            r = done[q]
        else:
            t = time.time()
            commits = answers[q].get("commits")
            g = (model.generate(answers[q]["prompt_ids"], cap, eos, forced=answers[q]["ids"]) if commits is None else
                 model.generate(answers[q]["prompt_ids"], cap, eos, forced=answers[q]["ids"], commits=commits))
            r = forced_row(q, answers[q], g, eos, time.time() - t, threads)
            out.write(json.dumps(r) + "\n")
            out.flush()
        rows.append(r)
        print(f"{q:5d} {r['len']:4d} {r['need']:4d} {r['flyx_rounds']:4d} {r['len'] / r['need']:6.3f} "
              f"{r['secs']:6.1f}", flush=True)
    return rows


def thread_counts(rows):
    """The thread counts rows were made at, as a report says them."""
    counts = sorted({r.get("threads") for r in rows}, key=str)
    return ", ".join("unrecorded" if t is None else str(t) for t in counts)


def against_free(rows, settings, path):
    """The forced rows against a free run's (dv_gsm8k.py) on the questions whose answers agree: (lines, passed)."""
    if settings.get("replay"):
        return [f"--free {path}: this run replayed the record's rounds (--replay), so its rounds are not a free "
                "run's"], False
    held, free = read_results(path)
    differ = differences({k: held.get(k) for k in SHARED}, {k: settings[k] for k in SHARED})
    if differ:
        return [f"--free {path}: made with other settings ({', '.join(differ)}); its rounds are not this run's"], False
    both = [r for r in rows if r["q"] in free]
    theirs = {free[r["q"]].get("threads") for r in both}
    other = [r for r in both if free[r["q"]].get("threads") not in (None, r.get("threads"))]
    if other:
        return [f"--free {path}: {len(other)} questions made at another thread count than here (first: question "
                f"{other[0]['q']}, {free[other[0]['q']]['threads']} threads against {other[0].get('threads')}); "
                "their rounds could differ at a near-tie"], False
    same = [r for r in both if free[r["q"]]["ids"] == r["ids"]]
    bad = [(r["q"], answer_rounds(free[r["q"]]["accepted"], r["len"]), answer_rounds(r["accepted"], r["len"]))
           for r in same]
    bad = [(q, f, r) for q, f, r in bad if f != r]
    lines = [f"against the free run {os.path.basename(path)} ({tree_name(held['drafter'])}; "
             + ("its thread count unrecorded, this run's " if None in theirs else "threads ") + thread_counts(both)
             + f"): {len(both)} questions in both, {len(same)} with the same answer; on them every round's a_k the "
             f"same in {len(same) - len(bad)}/{len(same)} questions ({sum(len(r['accepted']) for r in same)} rounds)"]
    for q, f, r in bad[:10]:
        k = next((i for i, (x, y) in enumerate(zip(f, r)) if x != y), min(len(f), len(r)))
        lines.append(f"  question {q}: the rounds part at forward {k + 1}: a_k {f[k:k + 3]} free, {r[k:k + 3]} forced")
    return lines, bool(same) and not bad


def replayed_rounds(row):
    """A replayed row's forwards that committed the record's counts: (its own a_k, the record's) each."""
    k = row["replayed"]
    return list(zip(row["accepted"][:k], row["committed"][:k]))


def against_record(rows):
    """A replayed run's own a_k against the record's rounds it committed, round for round: a report line."""
    per = [replayed_rounds(r) for r in rows]
    pairs = [p for rounds in per for p in rounds]
    same = sum(a == c for a, c in pairs)
    return (f"against the record's rounds: its own a_k the record's in {same} of {len(pairs)} rounds "
            f"({100.0 * same / max(len(pairs), 1):.2f}%); drafts accepted {sum(a for a, _ in pairs)} here, "
            f"{sum(c for _, c in pairs)} in the record; every round the same in "
            f"{sum(all(a == c for a, c in rounds) for rounds in per)} of {len(rows)} questions")


def tpf(lengths, needs):
    return f"{sum(lengths)}/{sum(needs)} = {sum(lengths) / sum(needs):.4f}" if sum(needs) else "-"


def mask_id(model_dir, token):
    """The MASK token's id in the target's tokenizer; refused when the tokenizer does not have it."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    mask = tok.convert_tokens_to_ids(token)
    if mask is None or (mask == tok.unk_token_id and token != tok.unk_token):
        sys.exit(f"--mask-token {token!r}: not a token of {model_dir}'s tokenizer")
    return mask


def assets(a):
    """The run's DvConfig, and a function that gives the target's and DV's weights (LlamaConfig, tensors): from a
    .flyx file with DV, or bifill-dv's checkpoints beside a Hugging Face directory."""
    if a.flyx:
        if a.canvas or a.selector or a.vlora:
            sys.exit("--flyx holds the target, the canvas and the selector: --canvas, --selector and --vlora go with "
                     "--model")
        cfg, W, header = from_flyx(a.flyx)
        return drafter_config(a, header, None, None), lambda: (cfg, W)
    if not (a.canvas and a.selector):
        sys.exit("--model needs --canvas and --selector")
    mask = mask_id(a.model, a.mask_token)
    with open(os.path.join(a.model, "config.json"), encoding="utf-8") as f:
        layers = json.load(f)["num_hidden_layers"]
    canvas, canvas_cfg = canvas_from_pt(a.canvas, layers)

    def weights():
        cfg, W = hf_llama(a.model)
        W.update(canvas)
        W.update(selector_from_pt(a.selector))
        if a.vlora:
            W.update(vlora_from_pt(a.vlora, cfg.layers))
        return cfg, W
    return drafter_config(a, None, canvas_cfg, mask), weights


def main(argv=None):
    a = arguments().parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.replay and a.free:
        sys.exit("--free compares a free run's rounds; a replayed run (--replay) commits the record's counts")
    try:
        eos, cap, answers = read_record(a.record, a.replay)
    except ValueError as e:
        sys.exit(str(e))
    qs = [q for q in sorted(answers) if a.first <= q < a.first + a.count]
    if not qs:
        sys.exit(f"{a.record} holds no prompt {a.first}..{a.first + a.count - 1}")
    t0 = time.time()
    dv, weights = assets(a)
    settings = run_settings(a, dv, eos, cap)
    threads = torch.get_num_threads()
    try:
        done, out = open_results(a.out, settings)
    except ValueError as e:
        sys.exit(str(e))
    todo = [q for q in qs if q not in done]
    why = threads_refused(a.out, done, threads) if todo else None
    if why:
        out.close()
        sys.exit(why)
    model = Dv(*weights(), dv, getattr(torch, a.dtype)) if todo else None
    del weights
    forced_to = os.path.basename(a.record) + ("'s answers and rounds" if a.replay else "")
    print(f"loaded in {time.time() - t0:.0f} s: {a.dtype}, {dv.slots} slots, tree {tree_name(dataclasses.asdict(dv))}, "
          f"fill {dv.fill}, mask {dv.mask_token}, eos {eos}, cap {cap}, {threads} threads; forced to "
          f"{forced_to}; {a.out} holds "
          f"{len(qs) - len(todo)} of the {len(qs)} questions", flush=True)
    rows = force(model, answers, qs, eos, cap, done, out, threads)
    out.close()
    del model

    forwards = sum(len(r["accepted"]) for r in rows)
    secs = sum(r["secs"] for r in rows)
    print(f"\n{len(rows)} questions ({qs[0]}..{qs[-1]}) at {thread_counts(rows)} threads: TPF "
          f"{tpf([r['len'] for r in rows], [r['need'] for r in rows])}; flyx's (new - 1) / rounds "
          f"{tpf([r['new'] - 1 for r in rows], [r['flyx_rounds'] for r in rows])}"
          + (" (the record's rounds)" if a.replay else "")
          + f"; {forwards} forwards in {secs:.0f} s ({secs / max(forwards, 1):.3f} s each), "
          f"{sum(sum(r['accepted']) for r in rows) / max(forwards, 1):.3f} drafts accepted a forward", flush=True)
    if a.replay:
        print(against_record(rows), flush=True)
    if a.free:
        lines, passed = against_free(rows, settings, a.free)
        print("\n".join(lines) + ("\nPASS" if passed else "\nFAIL"), flush=True)
        return 0 if passed else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
