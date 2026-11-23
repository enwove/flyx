# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""External benchmark scripts report successful samples and retain completed repetitions."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

from drafter_captures import stub


def invoke(script, env, *args, status=0):
    p = subprocess.run(["bash", script, *args], env=env, capture_output=True, text=True, timeout=30)
    assert p.returncode == status, (p.returncode, p.stdout, p.stderr)
    return p


def check_gguf(script, env):
    files = ["one model.gguf", "two.gguf"]
    p = invoke(script, env, *files)
    for name in ("one model", "two"):
        assert (name + " | flyx plain 12.3 tok/s | drafted 12.3 tok/s at 1.25 a round | "
                "prefill 12800 tok/s (512 tokens, 40.0 ms) | llama.cpp pp512 101.25 tg256 61.25") in p.stdout
    for case, status in (("plain-fail", 23), ("warm-fail", 24), ("llama-fail", 25), ("plain-empty", 1),
                         ("spec-empty", 1), ("warm-empty", 1), ("warm-zero", 1), ("llama-no-pp", 1), ("llama-no-tg", 1)):
        p = invoke(script, {**env, "CASE": case}, files[0], status=status)
        assert " | flyx plain" not in p.stdout and p.stderr, (case, p.stdout, p.stderr)


def check_server_exit(script, env, llama):
    stub(llama / "build/bin/llama-server", "echo 'fixture model load failed' >&2; exit 23")
    p = invoke(script, {**env, "PORT": "0"}, status=1)
    assert "did not become ready" in p.stderr
    logs = list((Path(env["TMPDIR"]) / "flyx-vs-llama").glob("run.*/llama_server_plain.log"))
    assert len(logs) == 1 and "fixture model load failed" in logs[0].read_text()


def main():
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="flyx-external-comparison-") as tmp:
        root = Path(tmp)
        bin_dir = root / "test tools"
        llama = root / "llama.cpp"
        bin_dir.mkdir()
        (llama / "build/bin").mkdir(parents=True)
        stub(bin_dir / "git", "echo fixture-revision")
        stub(bin_dir / "nvidia-smi", "echo fixture-gpu")
        stub(bin_dir / "flyx", '''
if [[ $1 == bench ]]; then
  case " $* " in *" --no-spec "*) mode=plain;; *) mode=spec;; esac
  [[ $CASE == "$mode-empty" ]] || echo 'decode: 12.3 tok/s, 1.25 tokens/round (cuda)'
  [[ $CASE != "$mode-fail" ]] || exit 23
else
  echo 'fixture reply'
  echo "[prompt 2 (reused 0) | prefill 1.0 ms | 2 tokens in 1.0 ms = $RATE tok/s | 1 rounds, 1.00 tokens/round]" >&2
fi''')
        stub(bin_dir / "python3", '''
if [[ $1 == *warm_prefill.py ]]; then
  case $CASE in
    warm-empty) echo 'no result';;
    warm-zero) echo '512 tokens; prefill ms: 0.0; median 0.0 (0K tok/s)';;
    *) echo '512 tokens; prefill ms: 40.0; median 40.0 (12.80K tok/s)';;
  esac
  [[ $CASE != warm-fail ]] || exit 24
else
  exec ''' + shlex.quote(sys.executable) + ''' "$@"
fi''')
        stub(llama / "build/bin/llama-bench", '''
[[ $CASE == llama-no-pp ]] || echo '| model | pp512 | 101.25 ± 0.4 |'
[[ $CASE == llama-no-tg ]] || echo '| model | tg256 | 61.25 ± 0.4 |'
[[ $CASE != llama-fail ]] || exit 25''')
        env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"], "BIN": str(bin_dir),
               "LLAMA": str(llama), "TMPDIR": tmp, "CASE": "ok", "N": "256"}
        env.pop("OUT", None)
        check_gguf(repo / "bench/external/gguf_vs_llama_cpp.sh", env)
        check_server_exit(repo / "bench/external/vs_llama_cpp.sh", env, llama)


if __name__ == "__main__":
    main()
