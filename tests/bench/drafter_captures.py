# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Drafter comparisons preserve existing results and fail on a bad sample or a different reply."""
import os
from pathlib import Path
import subprocess
import tempfile


def stub(path, body):
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


def check(script, root, bin_dir):
    out = root / ("previous results " + script.stem)
    out.mkdir()
    saved = out / "flyx_q4-plain.code.1.out"
    saved.write_bytes(b"an earlier run")
    env = {**os.environ, "OUT": str(out), "BIN": str(bin_dir), "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]}
    captures = {}
    for case, status in (("failed", 23), ("different", 1), ("retry", 0)):
        p = subprocess.run(["bash", script], env={**env, "CASE": case}, capture_output=True, text=True, timeout=15)
        assert p.returncode == status, (p.returncode, p.stdout, p.stderr)
        prefix = "Results in "
        run = Path(next(line[len(prefix):] for line in p.stdout.splitlines() if line.startswith(prefix)))
        assert run.parent == out and run not in captures and any(run.iterdir()), run
        assert saved.read_bytes() == b"an earlier run"
        for path, contents in captures.items():
            assert {f.name: f.read_bytes() for f in path.iterdir()} == contents
        captures[run] = {f.name: f.read_bytes() for f in run.iterdir()}


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="flyx-drafter-results-") as tmp:
        root = Path(tmp)
        bin_dir = root / "fake tools"
        bin_dir.mkdir()
        stub(bin_dir / "nvidia-smi", "echo fixture-gpu")
        stub(bin_dir / "git", "echo fixture-revision")
        stub(bin_dir / "flyx", """
if [[ $CASE == failed ]]; then exit 23; fi
if [[ $CASE == different && " $* " == *" --drafter dflash2 "* ]]; then echo different; fi
echo reply
echo 'round cost: 0=1.0 ms' >&2
echo '[prompt 1 (reused 0) | prefill 1.0 ms | 2 tokens in 1.0 ms = 12.0 tok/s | 1 rounds, 1.00 tokens/round]' >&2
while (( $# )); do
  if [[ $1 == --rounds ]]; then echo '{}' > "$2"; break; fi
  shift
done
""")
        for script in ("bench/drafters_27b.sh", "bench/replay/drafter_traces_27b.sh"):
            check(repo / script, root, bin_dir)
