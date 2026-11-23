# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Pass the documented K/V bit widths to the engine and reject malformed widths before opening a backend."""
import os
from pathlib import Path
import subprocess
import sys


exe, model = (Path(arg).resolve() for arg in sys.argv[1:])
env = {k: v for k, v in os.environ.items() if not k.upper().startswith("FLYX_")}


def run(method, backend="cpu_ref"):
    return subprocess.run([exe, "run", model, "--backend", backend, "--kv-method", method,
                           "--ctx", "128", "-p", "hello", "-n", "1"], env=env, capture_output=True, timeout=45)


exact = run("exact")
exact.check_returncode()
assert exact.stdout
for method in ("auto", "auto:2", "auto:3", "auto:04"):
    result = run(method)
    result.check_returncode()
    assert result.stdout == exact.stdout, method  # the CPU reference always reads every position

for method in ("auto:3.5", "strata:3bits", "strata:2:4", "auto:", "auto:1", "auto:5", "auto:" + "3" * 40):
    result = run(method, backend="absent")
    message = f"--kv-method {method}: 2, 3 or 4 bits".encode()
    assert result.returncode != 0 and message in result.stderr, (method, result.returncode, result.stderr)
