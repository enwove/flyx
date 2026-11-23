# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Generate a reply using an installed wheel or an unpacked release's own Python and native libraries."""
import os
from pathlib import Path
import sys


def main():
    package, model = (Path(arg).resolve(strict=True) for arg in sys.argv[1:])
    for key in list(os.environ):
        if key.upper().startswith("FLYX_"):
            del os.environ[key]
    sys.path.insert(0, str(package))
    import flyx

    if Path(flyx.__file__).resolve() != package / "flyx" / "__init__.py":
        raise RuntimeError("did not import the packaged binding")
    engine = flyx.Engine(model, backend="cpu_ref", max_ctx=128, speculative=False)
    if not Path(engine._lib._name).is_absolute():
        raise RuntimeError("the binding used the system library search path")
    session = engine.session()
    if len(session.generate(text="hello", max_new_tokens=2, ignore_eos=True)) != 2:
        raise RuntimeError("the packaged Python binding did not generate two tokens")
    session.close()
    engine.close()
    print("  ok   packaged Python binding generated with its bundled CPU backend")


if __name__ == "__main__":
    main()
