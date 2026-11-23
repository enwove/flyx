# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The DV reference (flyx_ref/dv.py) on GSM8K test questions: MiniCPM5-1B with bifill-dv's canvas and selector.

    python -I python/reference/dv_gsm8k.py MODEL_DIR --canvas CANVAS.pt --selector SELECTOR.pt \
        --questions gsm8k_test.jsonl [--prompts ar_test.jsonl] [--first 0] [--count 50] [--tree beam:31:8:cov] \
        [--slots 32] [--fill 0.9] [--max-new 512] [--dtype float32] [--vlora VLORA.pt] --out results.jsonl \
        [--ref REF.pt --label DVDT@0.9 [--pair OTHER.jsonl:LABEL]] [--transformers Q,Q] [--threads N]

The prompt is bifill-dv's: the question, then a newline and "Please reason step by step, and put your final answer
within \\boxed{}.", as the one user message of the model's chat template with thinking off (--prompts: bifill-dv's
data/ar_test.jsonl; every rendered prompt must equal the ids it records for that question).  An answer stops at its
first end token (kept) or at 512 tokens.  Checkpoints are read with torch.load(weights_only=True).

Per question: L (the answer's tokens), need (bifill's forwards: the first, all-MASK one included; ``rounds_needed``)
and flyx's rounds (``flyx_rounds``).  Totals: TPF = sum L / sum need (bifill's), flyx's sum (new - 1) / sum rounds
(new: the tokens before the end token), accuracy as bifill-dv scores it (only answers that ended count).

--out keeps the run: a first line with its settings (``run_settings``: the files by their sha256, the drafter's
configuration, the length cap, the dtype), then one JSON line a question (with the thread count it ran at).  A run
given a file with the same settings goes on after the questions it holds; a file with other settings, or whose rows
ran at another thread count, is refused, so that one file never holds two runs.  A last line cut off (a run stopped
while writing it) is dropped and its question runs again.

--ref (a bifill-dv combo.py result file: per_q[label] = {need, dis, correct}, outs[label] = the generated ids) puts
the reference's run `label` beside every question and compares the two runs (``compare``).  --pair OTHER.jsonl:LABEL
compares this run with another of ours on the questions both hold, and the reference's `label` with its LABEL there
(``pair``).  --transformers Q,Q decodes those questions again with transformers' own float32 greedy decoding and
compares the answers token for token.  A run whose --out already holds every question asked only reports again.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flyx_ref.dv import Dv, DvConfig, canvas_from_pt, hf_llama, selector_from_pt, vlora_from_pt  # noqa: E402

SUFFIX = "\nPlease reason step by step, and put your final answer within " + "\\" + "boxed{}."
MASK_TOKEN = "<unused_token_477>"
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def gsm_gold(answer):
    return answer.split("####")[-1].strip().replace(",", "")


def gsm_pred(text):
    """bifill-dv's reading of an answer (common.py gsm_pred): after </think> if any, the last number of the last
    boxed{...}, else of the text.  Its pattern's \\b is a word boundary, which never comes before the "oxed" of
    "\\boxed", so in effect it is the text's last number; kept as it is, so the accuracies compare."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    boxed = re.findall(r"\boxed\{([^}]*)\}", text)
    nums = NUMBER.findall(boxed[-1] if boxed else text)
    if not nums:
        return None
    v = nums[-1].replace(",", "")
    return v[:-3] if v.endswith(".00") else (v[:-2] if v.endswith(".0") else v)


def arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--canvas", required=True)
    ap.add_argument("--selector", required=True)
    ap.add_argument("--vlora")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--prompts")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--tree", default="beam:31:8:cov")
    ap.add_argument("--slots", type=int, default=0, help="canvas slots (0: the canvas's own)")
    ap.add_argument("--fill", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--dtype", default="float32", choices=("float64", "float32", "bfloat16"))
    ap.add_argument("--ref")
    ap.add_argument("--label", default="DVDT@0.9")
    ap.add_argument("--pair", help="OTHER.jsonl:LABEL, another run of ours and the reference's label for it")
    ap.add_argument("--transformers", default="", help="questions to decode again with transformers, e.g. 1,12")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=0)
    return ap


def drafter(a, canvas_cfg, mask_token):
    """The DV configuration a run uses: the options, and the canvas's own slots, top-k and mask."""
    return DvConfig(slots=a.slots or canvas_cfg["n"], mask_token=mask_token, fill=a.fill,
                    topk=canvas_cfg.get("res_k", 16), causal=bool(canvas_cfg.get("canvas_causal"))).set_tree(a.tree)


def tree_name(d):
    """A drafter configuration's tree as --tree spells it."""
    if d["tree"] == "beam":
        return f"beam:{d['width']}:{d['beam']}" + (":cov" if d["cov"] else "")
    return f"pool:{d['width']}:{d['pool_r']}:1"


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def model_digest(model_dir):
    """A model directory by what decides its answers: the name and sha256 of each of its .json, .jinja and
    .safetensors files (the configuration, the tokenizer and chat template, the weights)."""
    h = hashlib.sha256()
    for name in sorted(os.listdir(model_dir)):
        path = os.path.join(model_dir, name)
        if os.path.isfile(path) and name.endswith((".json", ".jinja", ".safetensors")):
            h.update(name.encode() + b"\0" + file_digest(path).encode() + b"\0")
    return h.hexdigest()


def run_settings(a, dv):
    """What decides a run's answers and counts, as --out keeps it.  The options that only choose questions, check or
    compare (--first, --count, --prompts, --ref, --label, --pair, --transformers), --out and --threads are not in it."""
    return {"model": model_digest(a.model), "questions": file_digest(a.questions), "canvas": file_digest(a.canvas),
            "selector": file_digest(a.selector), "vlora": file_digest(a.vlora) if a.vlora else None,
            "drafter": dataclasses.asdict(dv), "max_new": a.max_new, "dtype": a.dtype}


def differences(x, y):
    """The settings two runs differ in (a drafter's by field: drafter.tree, drafter.width, ...)."""
    flat = lambda s: {**{k: v for k, v in s.items() if k != "drafter"},  # noqa: E731
                      **{"drafter." + k: v for k, v in s.get("drafter", {}).items()}}
    fx, fy = flat(x), flat(y)
    return sorted(k for k in fx.keys() | fy.keys() if fx.get(k) != fy.get(k))


def read_lines(path):
    """A results file's JSON lines: (the objects, the bytes its whole lines take, whether a cut-off line follows them).
    A byte order mark (PowerShell's utf8 writes one) is read past, as json reads bytes.  A last line without its
    newline that is not JSON is the row of a run stopped while writing it; any other line that is not JSON is
    refused."""
    with open(path, "rb") as f:
        data = f.read()
    pos, objects, number = 0, [], 0
    while pos < len(data):
        end = data.find(b"\n", pos)
        end = len(data) if end < 0 else end + 1
        number += 1
        if data[pos:end].strip():
            try:
                objects.append(json.loads(data[pos:end]))
            except ValueError as e:
                if not data.endswith(b"\n") and end == len(data):
                    return objects, pos, True
                raise ValueError(f"{path}: line {number} is not JSON ({e})") from None
        pos = end
    return objects, pos, False


def settings_and_rows(path, lines):
    if not lines or "settings" not in lines[0]:
        raise ValueError(f"{path}: no settings line, so what made its rows is unknown; give another --out")
    return lines[0]["settings"], {r["q"]: r for r in lines[1:]}


def read_results(path):
    """A results file: (its settings, its rows by question).  A file without its settings line, or whose last line was
    cut off, is refused."""
    lines, _, cut = read_lines(path)
    if cut:
        raise ValueError(f"{path}: its last line is cut off (a run stopped while writing it); --out on it again drops "
                         "that line and runs its question again")
    return settings_and_rows(path, lines)


def open_results(path, settings):
    """--out: (the rows it already holds, by question; the file to add this run's rows to).  A new file starts with
    the settings; a file made with other settings is refused rather than mixed with this run.  A last line cut off
    (the run before stopped while writing it) is dropped, and its question runs again."""
    if os.path.exists(path) and os.path.getsize(path):
        lines, whole, cut = read_lines(path)
        held, done = settings_and_rows(path, lines)
        if held != settings:
            raise ValueError(f"{path} holds a run with other settings ({', '.join(differences(held, settings))}); "
                             f"give another --out")
        with open(path, "r+b") as f:
            if cut:
                f.truncate(whole)
                print(f"{path}: its last line was cut off (a run stopped while writing it): dropped, its question "
                      "runs again", file=sys.stderr, flush=True)
            f.seek(max(whole - 1, 0))
            if f.read(1) not in (b"\n", b""):
                f.write(b"\n")  # its last row written without its newline
        return done, open(path, "a", encoding="utf-8")
    out = open(path, "w", encoding="utf-8")
    out.write(json.dumps({"settings": settings}) + "\n")
    out.flush()
    return {}, out


def other_threads(done, threads):
    """The thread counts other than `threads` that the rows `done` were made with (None: a row that kept none).
    float32 and bfloat16 products round differently with another thread count, which can turn a tree at a near-tie,
    so one file's rows come from one count: a run that would add rows at another is refused."""
    return sorted({r.get("threads") for r in done.values()} - {threads}, key=str)


def threads_refused(path, done, threads):
    """open_results' refusal of a run at another thread count than the rows `path` holds: its message, or None."""
    other = other_threads(done, threads)
    if not other:
        return None
    counts = ", ".join("unrecorded" if t is None else str(t) for t in other)
    return (f"{path} holds rows made at other thread counts ({counts}) than this run's {threads}, whose rounds could "
            f"differ at a near-tie; run at their count (--threads) or give another --out")


def tpf(lengths, needs):
    return f"{sum(lengths)}/{sum(needs)} = {sum(lengths) / sum(needs):.4f}"


def compare(rows, ref, label, cap):
    """This run beside the reference's `label` on the same questions: its totals there; the answers both wrote the
    same and the forwards each needed for them; the answers cut at the length cap, and TPF on the questions neither
    run cut; the reference's own totals on every question it holds."""
    outs, per = ref["outs"][label], ref["per_q"][label]
    qs = [r["q"] for r in rows]
    L, need = sum(r["len"] for r in rows), sum(r["need"] for r in rows)
    rl, rn = sum(len(outs[q]) for q in qs), sum(per["need"][q] for q in qs)
    same = [r for r in rows if outs[r["q"]] == r["ids"]]
    print(f"reference {label} on the same questions: TPF {tpf([rl], [rn])}; "
          f"correct {sum(bool(per['correct'][q]) for q in qs)}/{len(qs)}; ours / reference = "
          f"{(L / need) / (rl / rn):.4f}; identical answers {len(same)}/{len(rows)}")
    if same:
        diff = sorted(r["need"] - per["need"][r["q"]] for r in same)
        print(f"on the {len(same)} identical answers ({sum(r['len'] for r in same)} tokens): forwards "
              f"{sum(r['need'] for r in same)} here, {sum(per['need'][r['q']] for r in same)} there; the same on "
              f"{diff.count(0)}/{len(same)} questions; differences here - there {diff}")
    uncut = [r for r in rows if r["len"] < cap and len(outs[r["q"]]) < cap]
    print(f"answers cut at {cap} tokens: {sum(r['len'] >= cap for r in rows)} here, "
          f"{sum(len(outs[q]) >= cap for q in qs)} there; on the {len(uncut)} questions neither cut: TPF "
          f"{tpf([r['len'] for r in uncut], [r['need'] for r in uncut])} here, "
          f"{tpf([len(outs[r['q']]) for r in uncut], [per['need'][r['q']] for r in uncut])} there")
    print(f"reference {label} on all its {len(outs)} questions: TPF {tpf([len(o) for o in outs], per['need'])}; "
          f"correct {sum(bool(c) for c in per['correct'])}/{len(outs)}")


def pair(rows, settings, path, ref, label, other_label, cap):
    """This run against another of ours (path) on the questions both hold: the answers both wrote the same and the
    forwards each needed for them; with the reference, its `label` against its `other_label` the same way."""
    held, other = read_results(path)
    both = [r for r in rows if r["q"] in other]
    same = [r for r in both if r["ids"] == other[r["q"]]["ids"]]
    a, b = sum(r["need"] for r in same), sum(other[r["q"]]["need"] for r in same)
    print(f"against {os.path.basename(path)} ({tree_name(held['drafter'])}; settings that differ: "
          f"{', '.join(differences(settings, held)) or 'none'}) on {len(both)} questions: identical answers "
          f"{len(same)}/{len(both)}" + (f"; on them forwards {a} here, {b} there ({a / b - 1:+.1%})" if same else ""))
    if ref:
        qs = [r["q"] for r in both]
        o1, o2 = ref["outs"][label], ref["outs"][other_label]
        n1, n2 = ref["per_q"][label]["need"], ref["per_q"][other_label]["need"]
        rs = [q for q in qs if o1[q] == o2[q]]
        a, b = sum(n1[q] for q in rs), sum(n2[q] for q in rs)
        print(f"reference {label} against {other_label} on those questions: identical answers {len(rs)}/{len(qs)}"
              + (f"; on them forwards {a} vs {b} ({a / b - 1:+.1%})" if rs else "")
              + f"; cut at {cap} tokens {sum(len(o1[q]) >= cap for q in qs)} vs {sum(len(o2[q]) >= cap for q in qs)};"
              f" TPF {tpf([len(o1[q]) for q in qs], [n1[q] for q in qs])} vs "
              f"{tpf([len(o2[q]) for q in qs], [n2[q] for q in qs])}")


def transformers_check(model_dir, prompts, rows, questions, eos, cap):
    """The answers to `questions` against transformers' AutoModelForCausalLM in float32, greedy (generate)."""
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    by_q = {r["q"]: r for r in rows}
    for q in questions:
        t = time.time()
        with torch.no_grad():
            out = hf.generate(torch.tensor([prompts[q]]), max_new_tokens=cap, do_sample=False, eos_token_id=list(eos),
                              pad_token_id=eos[0], top_p=None, temperature=None)[0, len(prompts[q]):].tolist()
        mine = by_q[q]["ids"]
        common = next((i for i, (x, y) in enumerate(zip(out, mine)) if x != y), min(len(out), len(mine)))
        print(f"question {q}: transformers' float32 greedy decoding {len(out)} tokens in {time.time() - t:.0f} s, "
              f"ours {len(mine)}; the same for the first {common}; identical: {out == mine}", flush=True)


def main():
    a = arguments().parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    items = [json.loads(line) for line in open(a.questions, encoding="utf-8")]
    qs = list(range(a.first, min(len(items), a.first + a.count)))
    recheck = [int(q) for q in a.transformers.split(",") if q]
    if any(q not in qs for q in recheck):
        sys.exit(f"--transformers {a.transformers}: only questions of this run ({qs[0]}..{qs[-1]})")
    prompts = {q: list(tok.apply_chat_template([{"role": "user", "content": items[q]["question"] + SUFFIX}],
                                               tokenize=True, add_generation_prompt=True, enable_thinking=False,
                                               return_dict=True)["input_ids"]) for q in qs}
    if a.prompts:
        recorded = {r["i"]: r["prompt"] for r in map(json.loads, open(a.prompts, encoding="utf-8"))}
        bad = [q for q in qs if prompts[q] != recorded[q]]
        if bad:
            sys.exit(f"rendered prompts differ from {a.prompts} for questions {bad}")
        print(f"prompts: {len(qs)}/{len(qs)} equal {os.path.basename(a.prompts)}'s", flush=True)
    eos = tuple(json.load(open(os.path.join(a.model, "generation_config.json")))["eos_token_id"])

    t0 = time.time()
    canvas, ccfg = canvas_from_pt(a.canvas, json.load(open(os.path.join(a.model, "config.json")))["num_hidden_layers"])
    dv = drafter(a, ccfg, tok.convert_tokens_to_ids(MASK_TOKEN))
    settings = run_settings(a, dv)
    try:
        done, out = open_results(a.out, settings)
    except ValueError as e:
        sys.exit(str(e))
    todo = [q for q in qs if q not in done]
    threads = torch.get_num_threads()
    why = threads_refused(a.out, done, threads) if todo else None
    if why:
        out.close()
        sys.exit(why)
    model = None
    if todo:
        cfg, W = hf_llama(a.model)
        W.update(canvas)
        W.update(selector_from_pt(a.selector))
        if a.vlora:
            W.update(vlora_from_pt(a.vlora, cfg.layers))
        model = Dv(cfg, W, dv, getattr(torch, a.dtype))
        del W
    del canvas
    print(f"loaded in {time.time() - t0:.0f} s: {a.dtype}, {dv.slots} slots, tree {tree_name(dataclasses.asdict(dv))}, "
          f"fill {dv.fill}, mask {dv.mask_token}, eos {eos}, {torch.get_num_threads()} threads; {a.out} holds "
          f"{sum(q in done for q in qs)} of the {len(qs)} questions", flush=True)

    ref = torch.load(a.ref, map_location="cpu", weights_only=True) if a.ref else None
    rows = []
    print(f"{'q':>5} {'L':>4} {'need':>4} {'flyx':>4} {'TPF':>6} {'secs':>6}"
          + (f" | ref {'L':>4} {'need':>4} {'TPF':>6} same" if ref else ""), flush=True)
    for q in qs:
        if q in done:
            r = done[q]
        else:
            t = time.time()
            g = model.generate(prompts[q], a.max_new, eos)
            ids = g["ids"]
            ended = ids[-1] in eos
            text = tok.decode(ids, skip_special_tokens=True)
            r = {"q": q, "len": len(ids), "need": g["need"], "flyx_rounds": g["flyx_rounds"],
                 "new": len(ids) - ended, "ended": ended,
                 "correct": ended and gsm_pred(text) == gsm_gold(items[q]["answer"]),
                 "accepted": g["accepted"], "ids": ids, "secs": time.time() - t, "threads": threads}
            out.write(json.dumps(r) + "\n")
            out.flush()
        rows.append(r)
        line = f"{q:5d} {r['len']:4d} {r['need']:4d} {r['flyx_rounds']:4d} {r['len'] / r['need']:6.3f} {r['secs']:6.1f}"
        if ref:
            ro, rn = ref["outs"][a.label][q], ref["per_q"][a.label]["need"][q]
            line += f" | ref {len(ro):4d} {rn:4d} {len(ro) / rn:6.3f} {'yes' if ro == r['ids'] else 'no'}"
        print(line, flush=True)
    out.close()
    del model

    fl_new, fl_rounds = sum(r["new"] - 1 for r in rows), sum(r["flyx_rounds"] for r in rows)
    forwards = [k for r in rows for k in r["accepted"]]
    secs = sum(r["secs"] for r in rows)
    total = tpf([r["len"] for r in rows], [r["need"] for r in rows])
    print(f"\n{len(rows)} questions ({qs[0]}..{qs[-1]}): TPF {total}; flyx's (new - 1) / rounds {fl_new}/{fl_rounds} = "
          f"{fl_new / fl_rounds:.4f}; correct "
          f"{sum(r['correct'] for r in rows)}/{len(rows)}; {len(forwards)} forwards in {secs:.0f} s "
          f"({secs / len(forwards):.3f} s each), {sum(forwards) / len(forwards):.3f} drafts accepted a forward; the "
          f"first forward accepts nothing in {sum(r['accepted'][0] == 0 for r in rows)}/{len(rows)}", flush=True)
    if ref:
        compare(rows, ref, a.label, a.max_new)
    if a.pair:
        path, _, other_label = a.pair.rpartition(":")
        pair(rows, settings, path, ref, a.label, other_label, a.max_new)
    if recheck:
        transformers_check(a.model, prompts, rows, recheck, eos, a.max_new)


if __name__ == "__main__":
    main()
