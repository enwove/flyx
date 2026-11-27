# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Exercise the release glibc gate through its command line."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile


CHECKER = Path(__file__).resolve().parents[2] / "packaging/check_glibc.py"


def run_check(*args, symbols=""):
    return subprocess.run([sys.executable, CHECKER, *args], input=symbols, text=True, capture_output=True)


def dynsym(*names):
    return "".join(f" {i}: 0000000000000000 0 FUNC GLOBAL DEFAULT UND {name} (3)\n"
                   for i, name in enumerate(names, start=1))


class GlibcTest(unittest.TestCase):
    def test_symbol_versions(self):
        valid = dynsym("fopen@GLIBC_2.2.5", "pipe2@GLIBC_2.9", "epoll_pwait2@GLIBC_2.35")
        result = run_check("-", symbols=valid)
        self.assertEqual(result.returncode, 0, result.stderr)
        bad = ["fmod@GLIBC_2.38", "arc4random@GLIBC_2.36"]
        result = run_check("-", symbols=valid + dynsym(*bad))
        self.assertNotEqual(result.returncode, 0)
        for symbol in bad:
            self.assertIn(symbol, result.stderr)
        self.assertNotIn("pipe2", result.stderr)
        self.assertIn("2.35", result.stderr)

    def test_override(self):
        result = run_check("--max-glibc", "2.9", "-", symbols=dynsym("epoll_pwait2@GLIBC_2.35"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("epoll_pwait2@GLIBC_2.35", result.stderr)

    def test_wheel_platforms(self):
        cases = [("manylinux_2_35_x86_64", True), ("manylinux_2_38_x86_64", False),
                 ("manylinux_2_17_x86_64.manylinux2014_x86_64", True),
                 ("manylinux_2_35_x86_64.manylinux_2_38_x86_64", False), ("linux_x86_64", False)]
        with tempfile.TemporaryDirectory() as tmp:
            for platform, valid in cases:
                with self.subTest(platform=platform):
                    wheel = Path(tmp) / f"flyx-1-py3-none-{platform}.whl"
                    with zipfile.ZipFile(wheel, "w") as archive:
                        archive.writestr("flyx/__init__.py", "")
                    result = run_check(wheel)
                    self.assertEqual(result.returncode == 0, valid, result.stderr)
                    if not valid:
                        self.assertIn(platform.split(".")[-1], result.stderr)

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires a Linux ELF and readelf")
    def test_nested_elf_in_package_and_repaired_wheel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            member = root / "plugins/helper"
            member.parent.mkdir()
            member.write_bytes(Path("/bin/ls").read_bytes())
            wheel = root / "flyx-1-py3-none-manylinux_2_5_x86_64.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.write(member, "flyx/_lib/helper")
            result = run_check("--max-glibc", "2.5", root, wheel)
            self.assertNotEqual(result.returncode, 0)
            self.assertRegex(result.stderr, r"plugins/helper: \S+@GLIBC_")
            self.assertRegex(result.stderr, r"flyx/_lib/helper: \S+@GLIBC_")


if __name__ == "__main__":
    unittest.main()
