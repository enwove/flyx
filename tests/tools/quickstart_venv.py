# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Exercise quickstart's pip recovery without a GPU, a network request or an existing environment."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def stub(path, body):
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


if __name__ == "__main__":
    source = Path(__file__).resolve().parents[2] / "tools/quickstart.sh"
    with tempfile.TemporaryDirectory(prefix="flyx-quickstart-") as tmp:
        root = Path(tmp)
        script = root / "checkout/tools/quickstart.sh"
        script.parent.mkdir(parents=True)
        shutil.copyfile(source, script)
        tools = root / "fake tools"
        tools.mkdir()
        stub(tools / "g++", "exit 0")
        stub(tools / "nvidia-smi", "echo 580.100")
        stub(tools / "nvcc", "echo fixture-compiler")
        stub(tools / "python3", "exit 91")  # neither case needs to create a new environment
        stub(tools / "curl", '[[ ${FAIL_FETCH:-} != 1 ]] || exit 37; echo fixture > "${!#}"')
        env = {**os.environ, "STEP": "deps", "PATH": str(tools) + os.pathsep + os.environ["PATH"],
               "MODELS": str(root / "models"), "CUDACXX": str(tools / "nvcc"), "TMPDIR": tmp}

        def run(venv, expected, **extra):
            p = subprocess.run(["bash", script], env={**env, "VENV": str(venv), **extra},
                               capture_output=True, text=True, timeout=10)
            assert p.returncode == expected, (p.returncode, p.stdout, p.stderr)

        foreign = root / "ordinary folder"
        foreign.mkdir()
        (foreign / ".original").write_text("user data")
        run(foreign, 1)
        assert list(foreign.iterdir()) == [foreign / ".original"] and (foreign / ".original").read_text() == "user data"

        venv = root / "environment with spaces"
        (venv / "bin").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("home = fixture")
        (venv / "bin/activate").write_text('export PATH="$VENV/bin:$PATH"\n')
        (venv / "get-pip.py").write_text("archived copy")
        stub(venv / "bin/python", """
if [[ $1 == -m && $2 == pip && $3 == --version ]]; then [[ -e "$VENV/has-pip" ]]; exit; fi
if [[ $1 == -m && $2 == pip && $3 == install ]]; then exit 0; fi
if [[ $1 == -m && $2 == ensurepip ]]; then exit 1; fi
[[ $(cat -- "$1") == fixture ]] && touch "$VENV/has-pip"
""")
        run(venv, 37, FAIL_FETCH="1")
        assert not (venv / "has-pip").exists()
        assert (venv / "get-pip.py").read_text() == "archived copy"
        run(venv, 0)
        assert (venv / "has-pip").exists() and (venv / "get-pip.py").read_text() == "archived copy"
