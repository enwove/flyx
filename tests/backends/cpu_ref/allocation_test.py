# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The CPU backend returns an allocation error to its caller, which can retry on the same engine."""
import os
from pathlib import Path
import resource
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))
import flyx


if __name__ == "__main__":
    model, library = map(Path, sys.argv[1:])
    engine = flyx.Engine(model, lib=library, backend_dir=library.parent, backend="cpu_ref",
                         max_ctx=32768, speculative=False)
    # Only this test process: hold its address space at its present size after loading the tiny model.
    limits = resource.getrlimit(resource.RLIMIT_AS)
    pages = int(Path("/proc/self/statm").read_text().split()[0])
    resource.setrlimit(resource.RLIMIT_AS, (pages * os.sysconf("SC_PAGE_SIZE"), limits[1]))
    try:
        try:
            engine.session()
        except flyx.FlyxError:
            pass
        else:
            raise AssertionError("the CPU session exceeded the process memory limit")
    finally:
        resource.setrlimit(resource.RLIMIT_AS, limits)
    engine.session().close()
    engine.close()
