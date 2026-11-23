# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""bifill-dv's per-question result files as check_dv.py's reference (REF.json), and their stored verdicts re-scored.

    python -I tests/e2e/dv_ref_json.py bifill-dv-ref/dyn1319_b2hcd.pt --list
    python -I tests/e2e/dv_ref_json.py bifill-dv-ref/dyn1319_b2hcd.pt -o dyn_b2h.json      # DVDT@0.9
    python -I tests/e2e/dv_ref_json.py bifill-dv-ref/dyn1319_orig.pt -o dyn_orig.json      # DVDT@0.9
    python -I tests/e2e/dv_ref_json.py bifill-dv-ref/dyn1319_orig.pt --label AR --ids -o ar.json  # plain decoding
    python -I tests/e2e/dv_ref_json.py bifill-dv/data/ar_test.jsonl --ids -o ar_test.json        # gen_ar's
    # check_dv.py's scoring against the verdicts the file stores: every run (--label: one); exit 1 if any differs
    python -I tests/e2e/dv_ref_json.py bifill-dv-ref/dyn1319_orig.pt --score --hf MiniCPM5-1B --gsm8k gsm8k_test.jsonl
    # how many of a run's answers are another's token for token (REF.json files written with --ids)
    python -I tests/e2e/dv_ref_json.py bifill-dv-ref/dyn1319_orig.pt --label AR --against ar_test.json

Three layouts are read, each {outs: {label: [the generated ids of each question]}, ...} once loaded:
  * bifill-dv's mega engine's (bifill/mega/engine.py, dp9_*.pt): perq[label] = {need, ok, ...}, cfg, runs[k].name
    (default label lossless@0.9);
  * dv_fused's (dyn1319_*.pt): per_q[label] = {need, correct, dis}, env, runs[k].label (default label DVDT@0.9);
  * gen_ar's JSONL (data/ar_test.jsonl): a row a question, by "i", with its answer's ids "resp", "correct", and the
    "response_text", "pred" and "gold" it was scored by: one run, plain decoding, labelled AR (need = L).
need is the forwards each answer took, counted as check_dv.py counts them (bifill-dv's).

REF.json:
  {"source": file name, "label": run, "settings": the file's cfg / env, "reported": the file's own summary of the run,
   "aggregate": {"questions", "tokens" (sum len), "need" (sum need), "tpf", "correct", "ended"},
   "questions": {"0": {"len": L, "need": forwards, "correct": bool[, "ids": the answer's first L tokens]}, ...}}
with L counted as check_dv.py counts it (tokens up to the first end token, included, at most 512).

The .pt files are pickles: torch.load(weights_only=True) only, and python -I.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I leaves the script's directory out

from check_dv import (EOS, MAX_NEW, answer_length, ended, gold_answers, gsm_correct, gsm_pred,  # noqa: E402
                      hf_decoder, identical_answers, read_json, read_jsonl)

DEFAULT_LABELS = ("lossless@0.9", "DVDT@0.9", "AR")  # each layout's lossless DV run; gen_ar's only run


def gen_ar(rows):
    """gen_ar's JSONL rows as the other layouts' result: one run, AR, its forwards its lengths."""
    rows = sorted(rows, key=lambda r: r["i"])
    if [r["i"] for r in rows] != list(range(len(rows))):
        raise ValueError("gen_ar rows: their 'i' are not 0 .. n-1")
    outs = [r["resp"] for r in rows]
    return {"outs": {"AR": outs},
            "per_q": {"AR": {"need": [answer_length(ids) for ids in outs], "correct": [r["correct"] for r in rows]}},
            "rows": rows}


def load(path):
    if path.endswith(".jsonl"):
        return gen_ar(read_jsonl(path))
    import torch
    return torch.load(path, weights_only=True, map_location="cpu")


def per_question(obj):
    return obj["per_q"] if "per_q" in obj else obj["perq"]


def verdicts(obj, label):
    """The file's own verdict on each answer of run `label`."""
    pq = per_question(obj)[label]
    return [bool(c) for c in (pq["correct"] if "correct" in pq else pq["ok"])]


def reference(obj, label, source, eos=EOS, max_new=MAX_NEW, ids=False):
    """REF.json's content for run `label` of a loaded result file (ids: with each answer's first L tokens)."""
    outs = obj["outs"][label]
    need = per_question(obj)[label]["need"]
    correct = verdicts(obj, label)
    if not (len(outs) == len(need) == len(correct)):
        raise ValueError(f"{source} [{label}]: {len(outs)} answers, {len(need)} counts, {len(correct)} scores")
    questions = {}
    for q, answer in enumerate(outs):
        L = answer_length(answer, eos, max_new)
        questions[str(q)] = {"len": L, "need": int(need[q]), "correct": correct[q]}
        if ids:
            questions[str(q)]["ids"] = [int(t) for t in answer[:L]]
    tokens = sum(r["len"] for r in questions.values())
    total_need = sum(r["need"] for r in questions.values())
    aggregate = {"questions": len(questions), "tokens": tokens, "need": total_need, "tpf": tokens / total_need,
                 "correct": sum(r["correct"] for r in questions.values()),
                 "ended": sum(ended(answer, eos) for answer in outs)}
    return {"source": source, "label": label, "settings": obj.get("cfg", obj.get("env")),
            "reported": reported_run(obj.get("runs", []), label, aggregate), "aggregate": aggregate,
            "questions": questions}


def reported_run(runs, label, aggregate):
    """The file's own summary of the run: by its label (dv_fused's), else (bifill-dv's mega engine names its runs
    otherwise, "mega sel lossless@0.9" for lossless@0.9) the one run whose TPF and correct answers are the ones counted
    here."""
    by_label = [r for r in runs if r.get("label") == label]
    if by_label:
        return by_label[0]
    same = [r for r in runs if round(r.get("tpf", 0), 4) == round(aggregate["tpf"], 4)
            and r.get("ok") == aggregate["correct"]]
    return same[0] if len(same) == 1 else None


def check_scores(obj, labels, golds, decode, source):
    """check_dv.py's scoring of each run's answers (decode: ids -> text) against the verdicts the file stores, and for
    gen_ar's rows the text, number and gold answer each was scored by: (report lines, how many differ in all)."""
    lines, differ_all = [], 0
    for label in labels:
        stored = verdicts(obj, label)
        texts = [decode(ids) for ids in obj["outs"][label]]
        ours = [gsm_correct(ids, text, golds[q]) for q, (ids, text) in enumerate(zip(obj["outs"][label], texts))]
        differ = sum(a != b for a, b in zip(ours, stored))
        differ_all += differ
        lines.append(f"{source} [{label}]: correct {sum(ours)} here, {sum(stored)} stored, {differ} differ")
        if "rows" in obj:  # gen_ar keeps no number for an answer that did not end
            rows = obj["rows"]
            text_differs = sum(t != r["response_text"] for t, r in zip(texts, rows))
            pred_differs = sum((gsm_pred(t) if ended(r["resp"]) else None) != r["pred"] for t, r in zip(texts, rows))
            gold_differs = sum(golds[r["i"]] != r["gold"] for r in rows)
            differ_all += text_differs + pred_differs + gold_differs
            lines.append(f"  {len(rows)} rows: text differs {text_differs}, pred differs {pred_differs}, gold differs "
                         f"{gold_differs}")
    return lines, differ_all


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("file", help="a bifill-dv result file (.pt, or gen_ar's .jsonl)")
    ap.add_argument("--label", help="the run (default: the file's lossless DV run, gen_ar's AR; --list names them)")
    ap.add_argument("--list", action="store_true", help="the file's runs and their TPF")
    ap.add_argument("-o", "--out", help="REF.json")
    ap.add_argument("--ids", action="store_true", help="REF.json keeps each answer's tokens (check_dv.py then counts "
                                                       "the answers that are the same token for token)")
    ap.add_argument("--score", action="store_true", help="re-score every run's answers (--label: one) by check_dv.py "
                                                         "against the file's own verdicts; needs --hf and --gsm8k")
    ap.add_argument("--hf", help="the model's Hugging Face directory: its tokenizer decodes the answers (--score)")
    ap.add_argument("--gsm8k", help="GSM8K's test set, {answer: '... #### N'} in question order (--score)")
    ap.add_argument("--against", nargs="+", default=[], metavar="REF.json",
                    help="how many of the run's answers are each REF.json's (written with --ids) token for token")
    args = ap.parse_args(argv)

    obj = load(args.file)
    source = os.path.basename(args.file)
    if args.list:
        for label in obj["outs"]:
            agg = reference(obj, label, source)["aggregate"]
            print(f"{label:24s} TPF {agg['tpf']:.4f} ({agg['tokens']} / {agg['need']}), correct {agg['correct']}, "
                  f"ended {agg['ended']}, mean length {agg['tokens'] / agg['questions']:.1f}")
        return 0
    if args.score:
        if not (args.hf and args.gsm8k):
            ap.error("--score needs --hf and --gsm8k")
        labels = [args.label] if args.label else list(obj["outs"])
        golds = gold_answers(args.gsm8k, max(len(obj["outs"][label]) for label in labels))
        lines, differ = check_scores(obj, labels, golds, hf_decoder(args.hf), source)
        print("\n".join(lines))
        return 1 if differ else 0
    label = args.label or next((d for d in DEFAULT_LABELS if d in obj["outs"]), None)
    if label not in obj["outs"]:
        sys.exit(f"error: {source} has no run {label!r}; it has {list(obj['outs'])}")
    ref = reference(obj, label, source, ids=args.ids or bool(args.against))
    agg = ref["aggregate"]
    print(f"{source} [{label}]: {agg['questions']} questions, TPF {agg['tpf']:.4f} ({agg['tokens']} / {agg['need']}), "
          f"correct {agg['correct']}, ended {agg['ended']}, mean length {agg['tokens'] / agg['questions']:.1f}")
    ours = {int(q): r["ids"] for q, r in ref["questions"].items()} if args.against else {}
    for path in args.against:
        other = read_json(path)
        if not all("ids" in r for r in other["questions"].values()):
            sys.exit(f"error: {path} has no answers' tokens (dv_ref_json.py --ids writes them)")
        same, n, prefix = identical_answers(ours, {int(q): r["ids"] for q, r in other["questions"].items()})
        print(f"  against {other['source']} [{other['label']}]: {same} of {n} answers identical token for token; "
              f"mean common prefix {prefix:.1f} tokens")
    if args.out:
        if not args.ids:
            for r in ref["questions"].values():
                r.pop("ids", None)
        with open(args.out, "w", encoding="utf8") as f:
            json.dump(ref, f, separators=(",", ":"))
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
