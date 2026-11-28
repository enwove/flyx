# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Parse every browser module without executing dashboard code."""
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    node = shutil.which("node")
    if not node:
        print("dashboard_syntax needs Node.js", file=sys.stderr)
        return 1 if os.environ.get("GITHUB_ACTIONS") == "true" else 77
    dashboard = Path(__file__).resolve().parents[3] / "apps/flyx-server/dashboard"
    modules = sorted(dashboard.rglob("*.js"))
    for module in modules:
        with module.open("rb") as source:
            result = subprocess.run([node, "--input-type=module", "--check"], stdin=source)
        if result.returncode:
            print(module.relative_to(dashboard), file=sys.stderr)
            return result.returncode
    print(f"Parsed {len(modules)} dashboard modules")
    return 0 if modules else 1


if __name__ == "__main__":
    sys.exit(main())
