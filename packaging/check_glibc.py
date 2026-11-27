# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Check Linux release ELF symbols and auditwheel platform tags against a glibc ceiling."""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


def version(value):
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", value):
        raise argparse.ArgumentTypeError(f"expected a numeric glibc version, got {value!r}")
    parts = [int(part) for part in value.split(".")]
    while len(parts) > 2 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def check_symbols(symbols, maximum):
    for name, required in re.findall(r"(\S+?@@?GLIBC_([0-9]+(?:\.[0-9]+)+))(?=\s|$)", symbols):
        if version(required) > maximum:
            yield name


def check_elf(path, maximum):
    result = subprocess.run(["readelf", "--dyn-syms", "--wide", path], capture_output=True,
                            encoding="utf-8", errors="replace")
    if result.returncode:
        raise ValueError(result.stderr.strip() or f"readelf exited with {result.returncode}")
    yield from check_symbols(result.stdout, maximum)


def check_wheel(path, maximum):
    legacy = {"manylinux1": "2.5", "manylinux2010": "2.12", "manylinux2014": "2.17"}
    for tag in path.stem.rsplit("-", 1)[-1].split("."):
        policy, _, arch = tag.partition("_")
        match = re.fullmatch(r"manylinux_([0-9]+)_([0-9]+)_\w+", tag)
        if match:
            required = ".".join(match.groups())
        elif policy in legacy and arch:
            required = legacy[policy]
        else:
            yield f"expected an auditwheel manylinux platform tag, got {tag}"
            continue
        if version(required) > maximum:
            yield tag

    # Repair may add libraries. Inspect the final wheel, including extensionless ELF files.
    with zipfile.ZipFile(path) as wheel, tempfile.TemporaryDirectory() as tmp:
        elf = Path(tmp) / "member"
        for entry in wheel.infolist():
            with wheel.open(entry) as src:
                header = src.read(4)
                if header != b"\x7fELF":
                    continue
                with elf.open("wb") as dst:
                    dst.write(header)
                    shutil.copyfileobj(src, dst)
            for problem in check_elf(elf, maximum):
                yield f"{entry.filename}: {problem}"


def check(path, maximum):
    if path.is_dir():
        for member in path.rglob("*"):
            if member.is_symlink() or not member.is_file():
                continue
            with member.open("rb") as src:
                if src.read(4) != b"\x7fELF":
                    continue
            for problem in check_elf(member, maximum):
                yield f"{member.relative_to(path)}: {problem}"
    elif path.suffix == ".whl":
        yield from check_wheel(path, maximum)
    else:
        yield from check_elf(path, maximum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-glibc", default="2.35", type=version, metavar="VERSION")
    parser.add_argument("paths", nargs="+", help="ELF, extracted package directory, wheel, or - for readelf output")
    args = parser.parse_args()
    failed = False
    ceiling = ".".join(map(str, args.max_glibc))
    for name in args.paths:
        try:
            problems = (check_symbols(sys.stdin.read(), args.max_glibc) if name == "-"
                        else check(Path(name), args.max_glibc))
            for problem in problems:
                print(f"{name} (max glibc {ceiling}): {problem}", file=sys.stderr)
                failed = True
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            print(f"{name}: inspection failed: {error}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
