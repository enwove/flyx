# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""DV's rounds chosen by SpecController, replayed on pinned runs' rounds: traces, sets and tables.

    python -I bench/replay/dv_choice.py traces bench A15.json A31.json OUT.txt [--limit N] [--note TEXT]
    python -I bench/replay/dv_choice.py traces run A15.jsonl A31.jsonl OUT.txt [--note TEXT]
    python -I bench/replay/dv_choice.py table DV_CHOICE [TRACES_DIR] [--rows all|rule|constants]

traces: a set's trace for the replay (bench/replay/dv_choice.cpp) from two pinned runs of the same text, DV of 15
drafts every round and of 31 (`flyx bench MODEL --drafter dv --width 15 ... --json A15.json`, its trace's tokens a
round; or `flyx run ... --drafter dv --width 15 --rounds A15.jsonl 2> A15.err`, its prompt's tokens read from the
.err beside it): a line a prompt, "prompt tokens ; tokens of each round of 15 ; tokens of each round of 31" (a
prompt whose two runs' prompts differ left out); --note: lines of `#` before them, what the runs were.

table: dv_choice (the build's bench target, DV_CHOICE its path) over TRACES_DIR's sets (default: the kept
traces, bench/replay/dv_traces) and the sets made from them: `rule`, each set on a new engine and
the chat set then GSM8K's first 200 in one (`mixed`, its GSM8K part); `constants`, the same with each of the rule's
constants set otherwise, one at a time (and the engine's probe credit full at load instead of 0).  Made from the
recorded sets: gsm200 (GSM8K's first 200 of gsm1319), inter (gsm200 with a chat request after every 8th, the chat
set's in turn), grow8 (2000 tokens after the 3110-token prompt at 8 a round of 15 and 9 a round of 31, where DV pays).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRACES = os.path.join(ROOT, "bench", "replay", "dv_traces")
SETS = ["gsm200", "gsm1319", "chat", "fs10", "fs4", "long", "inter", "grow8"]
# The rule's constants set otherwise, one at a time (the rule's own in the first column of the rows' names).
CONSTANTS = [
    ("kTreePrior", [4, 6, 12]), ("kGuessRounds", [4, 16]), ("kKeepCounts", [0.9, 0.97]),
    ("kLevelForget", [0.002, 0.01, 0.03]), ("kPayBack", [2, 4, 16]), ("kWidthTurn", [16, 32, 128]),
    ("kProbe", [16, 64]), ("kProbeRounds", [1, 4]), ("kProbeMax", [2, 16]), ("kProbeShare", [0.0025, 0.01]),
    ("kCreditCap", [1.0, 2.5]), ("credit", ["full"]),
]


def traces_bench(a: str, b: str, limit: int) -> list[tuple[int, list[int], list[int]]]:
    pa, pb = (json.load(open(p, encoding="utf-8"))["prompts"][:limit] for p in (a, b))
    return [(x["prompt_tokens"], [s[0] for s in x["trace"]], [s[0] for s in y["trace"]]) for x, y in zip(pa, pb)
            if x["prompt_tokens"] == y["prompt_tokens"]]


def traces_run(a: str, b: str) -> list[tuple[int, list[int], list[int]]]:
    def one(path: str) -> tuple[int, list[int]]:
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        err = open(os.path.splitext(path)[0] + ".err", encoding="utf-8", errors="replace").read()
        prompt = int(re.search(r"prompt (\d+) \(reused", err).group(1))
        return prompt, [r["tokens"] for r in rows[1:]]  # the rounds after the prefill's token

    (n, x), (m, y) = one(a), one(b)
    assert n == m, f"{a}: a prompt of {n} tokens, {b}: {m}"
    return [(n, x, y)]


def write(path: str, rows, note: str = "") -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        if note:
            f.write("".join(f"# {line}\n" for line in note.splitlines()))
        for n, x, y in rows:
            f.write(f"{n} ; {' '.join(map(str, x))} ; {' '.join(map(str, y))}\n")


def read(path: str) -> list[str]:
    return [line for line in open(path, encoding="utf-8").read().splitlines() if line and not line.startswith("#")]


def made(traces: str, out: str) -> None:
    """The recorded sets and the ones made from them, in out."""
    for name in ("gsm1319", "chat", "fs10", "fs4", "long"):
        with open(os.path.join(out, name + ".txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(read(os.path.join(traces, name + ".txt"))) + "\n")
    gsm200 = read(os.path.join(traces, "gsm1319.txt"))[:200]
    chat = read(os.path.join(traces, "chat.txt"))
    inter = []
    for i, line in enumerate(gsm200):
        inter.append(line)
        if i % 8 == 7:
            inter.append(chat[(i // 8) % len(chat)])
    sets = {"gsm200": gsm200, "inter": inter,
            "grow8": [f"3110 ; {' '.join(['8'] * 250)} ; {' '.join(['9'] * 223)}"]}
    for name, lines in sets.items():
        with open(os.path.join(out, name + ".txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")


LINE = re.compile(r"^(\S+)\s+([\d.]+) tok/s \| pinned 15\s+([\d.]+), 31\s+([\d.]+), plain\s+([\d.]+)")


def replay(binary: str, d: str, args: list[str], sets: list[str], each: bool = True) -> dict[str, list[float]]:
    out = subprocess.run([binary] + args + (["--each"] if each else []) + [os.path.join(d, s + ".txt") for s in sets],
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"dv_choice {' '.join(args)}: {out.stderr.strip()}")
    got = {}
    for line in out.stdout.splitlines():
        m = LINE.match(line)
        if m:
            got[os.path.splitext(m.group(1))[0] if m.group(1) != "all" else "all"] = [float(v) for v in m.groups()[1:]]
    return got


def row(binary: str, d: str, args: list[str]) -> list[float]:
    each = replay(binary, d, args, SETS)
    mixed = replay(binary, d, args, ["chat", "gsm200"], each=False)
    return [each[s][0] for s in SETS] + [mixed["gsm200"][0]]


def table(binary: str, traces: str, rows: str) -> int:
    with tempfile.TemporaryDirectory() as d:
        made(traces, d)
        heads = SETS + ["mixed"]
        constants = subprocess.run([binary, os.path.join(d, "grow8.txt")], capture_output=True, text=True).stdout
        print("\n".join(constants.splitlines()[:3]))  # the constants, the credit at load, the costs
        if rows in ("all", "rule"):
            each = replay(binary, d, [], SETS)
            mixed = replay(binary, d, [], ["chat", "gsm200"], each=False)
            print("\n| set | the rule | DV of 15 pinned | DV of 31 pinned | plain |\n|---|---:|---:|---:|---:|")
            for s in SETS:
                print(f"| {s} | " + " | ".join(f"{v:.1f}" for v in each[s]) + " |")
            print("| mixed (its GSM8K part) | " + " | ".join(f"{v:.1f}" for v in mixed["gsm200"]) + " |")
        if rows in ("all", "constants"):
            print("\n| tok/s | " + " | ".join(heads) + " |\n|---|" + "---:|" * len(heads))
            print("| the rule | " + " | ".join(f"{v:.1f}" for v in row(binary, d, [])) + " |")
            for name, values in CONSTANTS:
                for v in values:
                    print(f"| {name} {v} | " + " | ".join(f"{x:.1f}" for x in row(binary, d, [f"{name}={v}"])) + " |")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 5 and argv[0] == "traces" and argv[1] in ("bench", "run"):
        limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 1 << 30
        note = argv[argv.index("--note") + 1] if "--note" in argv else ""
        rows = traces_bench(argv[2], argv[3], limit) if argv[1] == "bench" else traces_run(argv[2], argv[3])
        write(argv[4], rows, note)
        return 0
    if len(argv) >= 2 and argv[0] == "table":
        traces = argv[2] if len(argv) > 2 and not argv[2].startswith("--") else TRACES
        rows = argv[argv.index("--rows") + 1] if "--rows" in argv else "all"
        return table(argv[1], traces, rows)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
