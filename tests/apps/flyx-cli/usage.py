# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The command line where it is asked something or cannot be followed: the version; the usage on stdout with status 0
when asked for; one line naming what is wrong (status 2) for an option it does not know; an error, never an abort, for
what a command cannot do.

    usage.py FLYX_EXE TINY_MODEL
"""
import os
from pathlib import Path
import re
import subprocess
import sys

exe, model = (Path(arg).resolve() for arg in sys.argv[1:])
env = {k: v for k, v in os.environ.items() if not k.upper().startswith("FLYX_")}
env["FLYX_BACKEND_DIR"] = str(exe.parent)


def run(*args):
    r = subprocess.run([exe, *args], env=env, capture_output=True, timeout=60, stdin=subprocess.DEVNULL)
    return r.returncode, r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace")


for flag in ("--version", "-V", "version"):
    rc, out, err = run(flag)
    assert rc == 0 and re.fullmatch(r"flyx \d+\.\d+\.\d+\s*", out) and not err, (flag, rc, out, err)

for args, begins in [(("--help",), "usage: flyx {run|"), (("-h",), "usage: flyx {run|"), (("help",), "usage: flyx {run|"),
                     (("run", "--help"), "usage: flyx {run|"), (("run", model, "--help"), "usage: flyx {run|"),
                     (("serve", "--help"), "usage: flyx serve"), (("monitor", "--help"), "usage: flyx monitor")]:
    rc, out, err = run(*args)
    assert rc == 0 and out.startswith(begins) and not err, (args, rc, out[:80], err[:200])

for args, says in [(("run", model, "-p", "hello", "--frobnicate"), "unknown option --frobnicate"),
                   (("run", model, "-p", "hello", "-n"), "-n needs a value"),
                   (("run", model, "-p", "hello", "more", "words"), "unexpected argument 'words'"),
                   (("frobnicate", model), "unknown command 'frobnicate'"),
                   (("run",), "run needs a model"),
                   (("serve", model, "--frobnicate"), "unknown option --frobnicate"),
                   (("serve", model, "--port"), "--port needs a value")]:
    rc, out, err = run(*args)
    assert rc == 2 and says in err and not out, (args, rc, out[:80], err[:300])
    assert err.count("\n") <= 2, (args, err)  # what is wrong, not the usage it would be lost in

for cmd in ("run", "chat", "template", "serve"):
    for value in ("", "{", "{} trailing", "[]", '"text"', "1", "null", '{"enable_thinking":"false"}'):
        rc, out, err = run(cmd, model, "--chat-template-kwargs", value)
        assert rc == 2 and "--chat-template-kwargs" in err and not out, (cmd, value, rc, out, err)
        assert ("JSON object" if "enable_thinking" not in value else "true or false") in err, err

# The monitor reads plain HTTP: an https URL is told so (the client's library throws on one).
rc, out, err = run("monitor", "https://127.0.0.1:9", "--once")
assert rc == 1 and "only http:// URLs" in err, (rc, out, err)
# A context no prompt fits in: 0 tokens at most, not a negative figure.
rc, out, err = run("run", model, "--backend", "cpu_ref", "--ctx", "1", "-p", "hello", "-n", "1")
assert rc == 1 and "at most 0 in a context of 1" in err, (rc, out, err)
print("usage: ok")
