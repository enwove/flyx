# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""bench/replay/dv_choice (the chooser's replay) on the kept traces (bench/replay/dv_traces): the sets made from
them, DV pinned and plain decoding at the speeds the traces give (they do not depend on the rule), a constant set on
the command line taking effect, the engine's credit full at load said so, and what is refused (an unknown name, a
missing set).

    python tests/bench/replay/dv_choice_test.py DV_CHOICE
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "bench", "replay"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "common"))

import dv_choice  # noqa: E402
from check import check, finish  # noqa: E402


def main() -> int:
    binary = sys.argv[1]
    with tempfile.TemporaryDirectory() as d:
        dv_choice.made(dv_choice.TRACES, d)
        lines = {s: len(dv_choice.read(os.path.join(d, s + ".txt"))) for s in dv_choice.SETS}
        check(lines == {"gsm200": 200, "gsm1319": 1319, "chat": 24, "fs10": 30, "fs4": 30, "long": 3, "inter": 225,
                        "grow8": 1}, f"the sets, a prompt a line: {lines}")
        # DV of 15 and of 31 every round and plain decoding on GSM8K's first 200, as the traces and the costs give them
        # whatever the rule.
        got = dv_choice.replay(binary, d, [], ["gsm200", "chat", "long"])
        check(got["gsm200"][1:] == [377.2, 341.9, 226.5] and got["chat"][1:] == [138.2, 120.6, 226.7] and
              got["long"][1:] == [93.8, 77.3, 210.6],
              f"pinned 15, 31 and plain decoding: the traces' speeds ({got['gsm200'][1:]}, {got['chat'][1:]}, "
              f"{got['long'][1:]})")
        # The other width's turns never due: no round of 31 (the means rate 15 the better at its cost).
        run = lambda *a: subprocess.run([binary, *a], capture_output=True, text=True)  # noqa: E731
        out = run("kWidthTurn=1000000", os.path.join(d, "gsm200.txt"))
        check(out.returncode == 0 and "kWidthTurn=1e+06" in out.stdout and ", 31 0," in out.stdout,
              "kWidthTurn=1000000: the constant in effect, no round of 31 (" +
              (out.stdout.splitlines()[-1] if out.stdout else out.stderr) + ")")
        out = run("credit=full", os.path.join(d, "grow8.txt"))
        check(out.returncode == 0 and "probe credit at load: full" in out.stdout, "credit=full: said so")
        out = run("kNoSuchThing=1", os.path.join(d, "gsm200.txt"))
        check(out.returncode == 2 and "not a SpecController constant" in out.stderr,
              "an unknown name refused: " + out.stderr.strip())
        out = run(os.path.join(d, "no_such_set.txt"))
        check(out.returncode == 1 and "cannot read" in out.stderr, "a missing set refused: " + out.stderr.strip())
    return finish("dv_choice_test")


if __name__ == "__main__":
    sys.exit(main())
