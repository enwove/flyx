# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The prompt sets DV's chosen rounds were measured and tuned on, written byte for byte as they ran.

    python -I bench/prompts/dv_sets.py GSM8K_TEST.jsonl OUT_DIR

GSM8K_TEST.jsonl: GSM8K's test split as bifill-dv keeps it (`data/gsm8k_test.jsonl`, 1319 lines of
{"question", "answer"}).  Each question is asked with bifill-dv's instruction after it.  Writes, one JSON object a line
(`flyx bench --prompts`), and checks each file's md5 against the sets that ran:

* fs4.jsonl, fs10.jsonl: GSM8K as a k-shot multi-turn chat (k = 4, 10): the shots are lines 1000 to 1000 + k - 1, each
  its question as the user's message and its answer as the assistant's (the `<<...>>` calculations taken out,
  `So the answer is \\boxed{N}.` for its `#### N`), then one of the first 30 questions ({"messages": [...]}).
* gsm30.jsonl: the first 30 questions ({"question": ...}); chat_then_gsm.jsonl: bench/prompts/chat_short.jsonl's
  24 lines, then those 30 (one engine: an engine whose first replies are not GSM8K's).
* gsm_long.jsonl: the first 30 questions as chat messages, the three summarize prompts (bench/prompts/
  summarize_*.txt: 3110, 3211 and 5949 tokens) after the 10th, 20th and 30th (long replies after GSM8K's).

fs4, fs10 and gsm_long ran with a backspace (U+0008) where `\\b` of `\\boxed` was meant, in the instruction and in the
shots' answers (their first generator wrote "\\boxed" in a Python string): kept, so that they are the bytes that
ran (BACKSPACED); and fs4 and fs10's lines end CR LF.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

BOXED = "\\boxed"
BACKSPACED = "\x08oxed"
SUFFIX = "\nPlease reason step by step, and put your final answer within " + BOXED + "{}."
PROMPTS = os.path.dirname(os.path.abspath(__file__))
MD5 = {
    "fs4.jsonl": "c59ddc5d39d21c9b09c41ad1d27fa29a",
    "fs10.jsonl": "5c1716996ad732f6d280e566642fe662",
    "gsm30.jsonl": "ec46101e9faa2f1731a78a93806ebcef",
    "chat_then_gsm.jsonl": "0eed39342111db49dbc6c04252b54216",
    "gsm_long.jsonl": "b4b30e6d539db50583f5c4134c911022",
}
SUMMARIZE = ("summarize_tree_rows.txt", "summarize_llama_qk_norms.txt", "summarize_tree_verification.txt")


def asked(question: str, boxed: str = BOXED) -> str:
    return question + SUFFIX.replace(BOXED, boxed)


def shot_answer(answer: str) -> str:
    body, final = re.sub(r"<<[^>]*>>", "", answer).rsplit("####", 1)
    return body.strip() + "\nSo the answer is " + BACKSPACED + "{" + final.strip() + "}."


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def sets(rows: list[dict]) -> dict[str, list[str]]:
    out = {}
    for k in (4, 10):
        shots = []
        for r in rows[1000:1000 + k]:
            shots.append(user(asked(r["question"], BACKSPACED)))
            shots.append({"role": "assistant", "content": shot_answer(r["answer"])})
        out[f"fs{k}.jsonl"] = [json.dumps({"messages": shots + [user(asked(r["question"], BACKSPACED))]})
                               for r in rows[:30]]
    gsm30 = [json.dumps({"question": asked(r["question"])}) for r in rows[:30]]
    out["gsm30.jsonl"] = gsm30
    chat = open(os.path.join(PROMPTS, "chat_short.jsonl"), encoding="utf-8").read().splitlines()
    out["chat_then_gsm.jsonl"] = chat + gsm30
    long = []
    for i, r in enumerate(rows[:30]):
        long.append(json.dumps({"messages": [user(asked(r["question"], BACKSPACED))]}))
        if i % 10 == 9:
            text = open(os.path.join(PROMPTS, SUMMARIZE[i // 10]), encoding="utf-8", newline="").read()
            long.append(json.dumps({"messages": [user(text)]}))
    out["gsm_long.jsonl"] = long
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
    os.makedirs(sys.argv[2], exist_ok=True)
    bad = 0
    for name, lines in sets(rows).items():
        end = "\r\n" if name.startswith("fs") else "\n"  # fs4 and fs10 as first written on Windows, in text mode
        data = "".join(line + end for line in lines).encode("utf-8")
        with open(os.path.join(sys.argv[2], name), "wb") as f:
            f.write(data)
        md5 = hashlib.md5(data).hexdigest()
        same = md5 == MD5[name]
        bad += not same
        print(f"{name}: {len(lines)} prompts, md5 {md5} " +
              ("(as they ran)" if same else f"(they ran as {MD5[name]})"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
