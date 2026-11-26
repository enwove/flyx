# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""A test script's checks, as tests/common/check.h is a C++ test's: each one printed, ok or FAIL (a failure with where
it is in the test and, when the check gives it, what it got), the failures counted, and the script's verdict and exit
code from the count.

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
    from check import check, finish  # noqa: E402

    check(reply["done"], "the request finishes", reply)  # FAIL tests/apps/x_test.py:52: the request finishes: got {...}
    sys.exit(finish("x_test"))                            # x_test: PASSED (12 checks), exit code 0; FAILED ..., 1
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
checks = 0
failures = 0


def _where(frame) -> str:
    """The caller's file (its path in the repository when it is in it) and line."""
    path = os.path.abspath(frame.f_code.co_filename)
    if os.path.normcase(path).startswith(os.path.normcase(ROOT) + os.sep):
        path = path[len(ROOT) + 1:].replace(os.sep, "/")
    return f"{path}:{frame.f_lineno}"


def check(ok, what: str, got=None) -> None:
    """Counts the check and prints it; `got`, when not None, is shown on a failure."""
    global checks, failures
    checks += 1
    if ok:
        print(f"ok   {what}", flush=True)
        return
    failures += 1
    print(f"FAIL {_where(sys._getframe(1))}: {what}" + ("" if got is None else f": got {got!r}"), flush=True)


def finish(name: str) -> int:
    """The verdict line, and the exit code: 0 when no check failed, else 1."""
    if failures:
        print(f"{name}: FAILED ({failures} failures, {checks} checks)", flush=True)
    else:
        print(f"{name}: PASSED ({checks} checks)", flush=True)
    return 1 if failures else 0
